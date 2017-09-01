from multiprocessing import Pool
from http.server import BaseHTTPRequestHandler, HTTPStatus
from socketserver import ThreadingMixIn, TCPServer

import os
import threading
import traceback
import urllib
import json

import webcontent
import describefloat

# main work process for worker pool
def do_work(w, p, s, show_format):
    try:
        s = describefloat.explain_all(s, w, p, show_format=show_format)
        return True, s
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
    fmt_cache = None
    demo_cache = None
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
            s = ''

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
            show_format = path in {'fmt', 'jfmt'}
            the_pool = type(self).the_pool

            if show_format:
                the_cache = type(self).fmt_cache
                w, p, s = args
                kargs = (w, p,)
            else:
                the_cache = type(self).demo_cache
                kargs = args

            # get cached content
            cached = the_cache.lookup(kargs)
            if cached is None:
                success, cached = the_pool.apply(do_work, (*args, show_format))
                if success:
                    the_cache.update(kargs, cached)

            # construct page
            body = (webcontent.skeleton_indent + '{}\n\n<br>').format(webcontent.create_webform(*args))
            body.replace('\n', '\n' + webcontent.skeleton_indent)
            body += '\n\n' + webcontent.pre(cached)

            return webcontent.skeletonize(body), 'text/html'

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


ncores = os.cpu_count()
default_pool_size = max(1, min(ncores - 1, (ncores // 2) + 1))

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_cache', type=int, default=10000,
                        help='number of demos to cache')
    parser.add_argument('--fmt_cache', type=int, default=1000,
                        help='number formats to cache')
    parser.add_argument('--workers', type=int, default=default_pool_size,
                        help='number of worker processes to run in parallel')

    parser.add_argument('--host', type=str, default='localhost',
                        help='server host')
    parser.add_argument('--port', type=int, default=8000,
                        help='server port')
    args = parser.parse_args()

    fmt_cache = AsyncCache(args.fmt_cache)
    demo_cache = AsyncCache(args.demo_cache)
    with Pool(args.workers, maxtasksperchild=100) as the_pool:
        class MyHTTPRequestHandler(AsyncHTTPRequestHandler):
            fmt_cache = fmt_cache
            demo_cache = demo_cache
            the_pool = the_pool

            print('caching {:d} fmt, {:d} demo'.format(args.fmt_cache, args.demo_cache))
            print('{:d} worker processes'.format(args.workers))

        with ThreadedTCPServer((args.host, args.port,), MyHTTPRequestHandler) as server:
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
