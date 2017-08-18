from multiprocessing import Pool
from http.server import BaseHTTPRequestHandler, HTTPStatus
from socketserver import ThreadingMixIn, TCPServer

import os
import time
import threading
import urllib

# enc

page = '''<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <title>aserver</title>
  </head>

  <body>
    <header>
      <h1>aserver</h1>
      <aside>test site plz ignore</aside>
    </header>

    <br>
    <pre>
{}
    </pre>
  </body>
</html>
'''

def do_work(mode, w, p, s):
    mypid = os.getpid()
    sleep_ms = int(w)

    # begin work
    start = time.time()

    time.sleep(3.0 + (sleep_ms / 1000))

    content = ('got some args:\n  mode: {}\n  w   : {}\n  p   : {}\n  s   : {}'
               .format(repr(mode), repr(w), repr(p), repr(s)))

    # end work
    elapsed = time.time() - start

    return content + '\nelapsed {:2f}s on pid {:d}'.format(elapsed, mypid)

# simple threadsafe key-value store with LRU
class AsyncCache(object):

    def __init__(self, n = 1024):
        self.n = int(n)
        assert self.n > 0

        self.lock = threading.Lock()
        # key : [value, indexof(value, self.order)]
        self.cache = {}
        # ordering for LRU
        self.order = []

    def lookup(self, k):
        with self.lock:
            record = self.cache.get(k, None)
            if record is None:
                # cache doesn't have it
                return None
            else:
                v, idx = record[0], record[1]
                # update LRU order
                print('lookup', self.order)
                self.order.pop(idx)
                self.order.append(k)
                print(self.order)
                # update record in cache
                record[1] = len(self.order) - 1
                # TODO: oops, lru isn't this easy...
                return v

    def update(self, k, v):
        with self.lock:
            existing_record = self.cache.get(k, None)
            if existing_record is None:
                # LRU: delete an item if necessary
                if len(self.order) >= self.n:
                    print('del', self.order)
                    xk = self.order.pop(0)
                    self.cache.pop(xk)
                    print(self.order)
                # add new record
                record = [v, len(self.order)]
                self.cache[k] = record
                print('new', self.order)
            else:
                existing_record[0] = v
                print('update', self.order)
                self.order.pop(existing_record[1])
                print(self.order)
                existing_record[1] = len(self.order) - 1
            self.order.append(k)
            print(self.order)
            print(self.cache)

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
        self.wfile.write(content)

    # helpers

    def translate_path(self):
        pr = urllib.parse.urlparse(self.path)

        args = {}
        for f in pr.query.split('&'):
            name_arg = f.split('=')
            if len(name_arg) == 2:
                name = urllib.parse.unquote_plus(name_arg[0].strip())
                if len(name) > 0 and name not in args:
                    arg = urllib.parse.unquote_plus(name_arg[1].strip())
                    args[name] = arg

        return pr.path, args

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
        myself = threading.current_thread().name
        path, args = self.translate_path()
        processed_args = self.process_args(args)

        # get cached content
        cached = type(self).the_cache.lookup(processed_args)
        if cached is None:
            cached = type(self).the_pool.apply(do_work, (path, *processed_args))
            type(self).the_cache.update(processed_args, cached)

        s = '\n'.join(
            ('using {}'.format(repr(myself)),
             repr(path),
             *('  {} : {}'.format(repr(k), repr(v)) for k, v in args.items()),
             '',
             cached,
             '',
             repr(type(self).the_cache.cache),
             repr(type(self).the_cache.order),)
            )
        return page.format(s)

    # also returns the content that would have been sent for these headers
    def send_head(self):
        content = bytes(self.construct_content(), 'utf-8')
        ctype = 'text/html'

        try:
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-type', ctype)
            self.send_header('Content-Length', len(content))
            self.end_headers()
            return content
        except:
            # could do cleanup like closing a file
            raise


class ThreadedTCPServer(ThreadingMixIn, TCPServer):
    allow_reuse_address = True


HOST = 'localhost'
PORT = 8000

if __name__ == '__main__':
    import sys

    the_cache = AsyncCache(3)
    with Pool(2) as the_pool:
        class MyHTTPRequestHandler(AsyncHTTPRequestHandler):
            the_cache = the_cache
            the_pool = the_pool

        with ThreadedTCPServer((HOST, PORT,), MyHTTPRequestHandler) as server:
            server_thread = threading.Thread(target=server.serve_forever)
            server_thread.daemon = True
            server_thread.start()

            print('server on thread:', server_thread.name)
            print('close stdin to stop.')

            for line in sys.stdin:
                pass

            print('stdin closed, stopping.')
            server.shutdown()
            print('goodbye!')
