from multiprocessing import Pool
from http.server import BaseHTTPRequestHandler, HTTPStatus
from socketserver import ThreadingMixIn, TCPServer

import os
import time
import threading
import traceback
import urllib
import json

import webcontent

# for testing purposes
def sleepfor(x):
    mypid = os.getpid()
    start = time.time()
    time.sleep(x)
    elapsed = time.time() - start
    return 'elapsed {:2f}s on pid {:d}'.format(elapsed, mypid)

# main work targets for protocols

def do_work(mode, w, p, s):
    try:
        sleep_s = 3.0 + int(w) / 1000

        content = ('{} mode:\n  w: {}\n  p: {}\n  s: {}'
                   .format(repr(mode.upper()), repr(w), repr(p), repr(s)))
        sleepy = sleepfor(sleep_s)

        return True, content + '\n' + sleepy
    except Exception as e:
        s = ('Caught {} while working.\n\n{}'
             .format(repr(e), traceback.format_exc()))
        return False, s

# for LRU's circular DLL
_PREV, _NEXT, _KEY = 0, 1, 2

# simple threadsafe key-value store with LRU
class AsyncCache(object):

    def __init__(self, n = 1024):
        self.n = int(n)
        assert self.n > 0
        self.lock = threading.Lock()
        self.reset()

    # not threadsafe - only call if you have the lock!
    def _move_to_front(self, link):
        link_prev, link_next, link_k = link
        # update LRU order, if necessary
        if link_next is not self.root:
            # remove link from list
            link_prev[_NEXT] = link_next
            link_next[_PREV] = link_prev
            # get the end of the list
            last = self.root[_PREV]
            # insert this link between the end and self.root
            link[_NEXT] = self.root
            self.root[_PREV] = link
            last[_NEXT] = link
            link[_PREV] = last

    # user-facing cache interface

    def reset(self):
        with self.lock:
            # key : (value, link)
            self.cache = {}
            # ordering for LRU, stored in circular doubly-linked listed
            self.root = []
            self.root[:] = (self.root, self.root, None,)

    def lookup(self, k):
        with self.lock:
            record = self.cache.get(k, None)
            if record is None:
                # cache doesn't have it
                return None
            else:
                v, link = record
                self._move_to_front(link)
                return v

    def update(self, k, v):
        with self.lock:
            record = self.cache.get(k, None)
            if record is None:
                # insert a new element
                if len(self.cache) < self.n:
                    last = self.root[_PREV]
                    link = [last, self.root, k]
                    last[_NEXT] = link
                    self.root[_PREV] = link
                    self.cache[k] = (v, link,)
                # at capacity - move root to reclaim an existing slot in the DLL
                else:
                    # root becomes new link
                    link = self.root
                    link[_KEY] = k
                    # next list becomes new root
                    self.root = link[_NEXT]
                    old_k = self.root[_KEY]
                    self.root[_KEY] = None
                    # update cache
                    del self.cache[old_k]
                    self.cache[k] = (v, link,)
            else:
                old_v, link = record
                # the key stays the same, move the existing one to the front
                self._move_to_front(link)
                # but we need to update the value in the cache
                self.cache[k] = (v, link,)

    # serialization with json

    def to_json(self):
        with self.lock:
            # make a list of (key, value,) pairs in the order things were used (oldest first)
            jl = []
            link = self.root[_NEXT]
            while link is not self.root:
                _, link_next, k = link
                record = self.cache.get(k, None)
                if record is not None:
                    v, link = record
                    jl.append((k, v,))
                link = link_next
            return json.dumps(jl)

    def from_json(self, s):
        # adds a list of (k, value,) pairs to the cache in order
        jl = json.loads(s)
        for k, v in jl:
            self.update(tuple(k), v)


class AsyncHTTPRequestHandler(BaseHTTPRequestHandler):

    # subclass and overwrite these before using
    the_cache = None
    the_pool = None

    # interface

    server_version = 'aserver/0.0'

    def do_HEAD(self):
        self.send_head()

    def do_GET(self):
        content = self.send_head()
        if content is not None:
            try:
                self.wfile.write(content)
            except Exception:
                # could try again, but we don't really care that much
                pass

    # helpers

    def translate_path(self):
        pr = urllib.parse.urlparse(self.path)
        path = pr.path.lower()
        if webcontent.empty_re.fullmatch(path):
            path = webcontent.root_page

        page_match = webcontent.page_re.fullmatch(path)
        if page_match:
            return page_match.group(1), None

        protocol_match = webcontent.protocol_re.fullmatch(path)
        if protocol_match:
            args = {}
            for f in pr.query.split('&'):
                name_arg = f.split('=')
                if len(name_arg) == 2:
                    name = urllib.parse.unquote_plus(name_arg[0].strip())
                    if len(name) > 0 and name not in args:
                        arg = urllib.parse.unquote_plus(name_arg[1].strip())
                        args[name] = arg
            return protocol_match.group(1), args

        return None, None

    def process_args(self, args):
        try:
            w = int(args['w'])
            assert w > 2
        except Exception:
            w = 8

        try:
            p = int(args['p'])
            assert p > 2
        except Exception:
            p = 24

        try:
            s = str(args['s'])
        except Exception:
            s = 'the default'

        return w, p, s

    def construct_content(self):
        path, raw_args = self.translate_path()

        # no content
        if path is None:
            return None, None

        # static page
        elif raw_args is None:
            return webcontent.page_content[path]

        # dynamic content protocol
        else:
            args = self.process_args(raw_args)
            myself = threading.current_thread().name

            # get cached content
            cached = type(self).the_cache.lookup(args)
            if cached is None:
                success, cached = type(self).the_pool.apply(do_work, (path, *args))
                if success:
                    type(self).the_cache.update(args, cached)

            # stress test cache serialization
            cache_json = type(self).the_cache.to_json()
            type(self).the_cache.reset()
            type(self).the_cache.from_json(cache_json)

            # format stuff
            s = '\n'.join(
                ('using {}'.format(repr(myself)),
                 repr(path),
                 *('  {} : {}'.format(repr(k), repr(v)) for k, v in raw_args.items()),
                 '',
                 cached,
                 '',
                 type(self).the_cache.to_json(),)
                )

            return webcontent.skeletonize(webcontent.pre(s)), 'text/html'

    # also returns the content that would have been sent for these headers
    def send_head(self):
        try:
            content, ctype = self.construct_content()
        except Exception as e:
            s = ('Caught {} while preparing content.\n\n{}'
                 .format(repr(e), traceback.format_exc()))
            content = webcontent.skeletonize(webcontent.pre(s))
            ctype = 'text/html'

        try:
            if content is None or ctype is None:
                self.send_error(HTTPStatus.NOT_FOUND, 'Unknown request')
                return None
            else:
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', len(content))
                self.end_headers()
                return content
        except:
            # could do cleanup like closing a file
            raise


class ThreadedTCPServer(ThreadingMixIn, TCPServer):
    allow_reuse_address = True
    daemon_threads = True


HOST = 'localhost'
PORT = 8000

cache_size = 10000
ncores = os.cpu_count()
pool_size = max(1, min(ncores - 1, (ncores // 2) + 1))

if __name__ == '__main__':
    import sys

    the_cache = AsyncCache(cache_size)
    with Pool(pool_size) as the_pool:
        class MyHTTPRequestHandler(AsyncHTTPRequestHandler):
            the_cache = the_cache
            the_pool = the_pool

            print('caching {:d} requests'.format(cache_size))
            print('{:d} worker processes'.format(pool_size))

        with ThreadedTCPServer((HOST, PORT,), MyHTTPRequestHandler) as server:
            server_thread = threading.Thread(target=server.serve_forever)
            server_thread.daemon = True
            server_thread.start()

            print('server on thread:', server_thread.name)
            print('close stdin to stop.')

            for line in sys.stdin:
                pass

            print('stdin closed, stopping.')
            the_pool.close()
            print('workers closing...')
            the_pool.join()
            print('workers joined successfully.')
            server.shutdown()
            print('goodbye!')
