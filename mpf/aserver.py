from multiprocessing import Pool
from http.server import BaseHTTPRequestHandler, HTTPStatus
from socketserver import ThreadingMixIn, TCPServer

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

def do_work(args):
    return ''

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
                self.order.append(self.order.pop(idx))
                # update record in cache
                record[1] = len(self.order) - 1
                return v

    def update(self, k, v):
        # placeholder index
        record = [v, -1]
        with self.lock:
            self.cache[k] = record
            self.order.append(k)
            # LRU: delete an item if necessary
            if len(self.order) > self.n:
                self.cache.pop(self.order.pop(0))
            # fix index
            record[1] = len(self.order) - 1
        

class AsyncHTTPRequestHandler(BaseHTTPRequestHandler):

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

    def construct_content(self):
        myself = threading.current_thread().name
        path, args = self.translate_path()
        s = '\n'.join(
            ('using {}'.format(repr(myself)),
             repr(path),
             *('  {} : {}'.format(repr(k), repr(v)) for k, v in args.items()),)
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

    with ThreadedTCPServer((HOST, PORT,), AsyncHTTPRequestHandler) as server:
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
