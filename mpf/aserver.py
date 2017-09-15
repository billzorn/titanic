# aserver: parallel webserver and LRU cache in one file

import sys
import os
import threading
import traceback
import urllib
import json
from http.server import BaseHTTPRequestHandler, HTTPStatus
from socketserver import ThreadingMixIn, TCPServer

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
                raise KeyError(k)
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

ERROR_MESSAGE = '''\
<html>
<head>
<title>%(code)d %(message)s</title>
</head>
<body>


<center>
<h1>%(code)d %(message)s</h1>
<p>{} {}</p>
</center>
<hr>


<pre>%(explain)s</pre>


</body>
</html>
'''
ERROR_CONTENT_TYPE = 'text/html'

class AsyncHTTPRequestHandler(BaseHTTPRequestHandler):
    server_version = 'aserver/0.1'
    sys_version = "Python/" + sys.version.split()[0]
    protocol_version = 'HTTP/1.0'
    error_message_format = ERROR_MESSAGE.format(server_version, sys_version)
    error_content_type = ERROR_CONTENT_TYPE
    traceback_log = True
    traceback_send = True

    # These do not need to be overridden again; they just call send_head.
    def do_HEAD(self):
        self.send_head()
    def do_GET(self):
        content = self.send_head()
        if content is not None:
            try:
                self.wfile.write(content)
            except Exception as e:
                self.log_error('Caught {} while writing response to GET.\n\n{}',
                               repr(e), traceback.format_exc())

    # This does not need to be overridden again: it just calls self.construct_content().
    # Override that in subclasses to change the content produced by the handler.
    def send_head(self):
        try:
            response, msg, headers, content = self.construct_content()

        except Exception as e:
            code = HTTPStatus.INTERNAL_SERVER_ERROR
            message = None
            explain = None

            if self.traceback_log or self.traceback_send:
                explain_traceback = ('Caught {} while preparing content.\n\n{}'
                                     .format(repr(e), traceback.format_exc()))
                if self.traceback_send:
                    message = type(e).__name__
                    explain = explain_traceback

            self.send_error(code, message, explain)

            if self.traceback_log:
                self.log_message('%s', explain_traceback)

            # send_error already writes the body, so we don't need to return anything
            return None

        else:
            self.send_response(response, msg)
            for k, v in headers:
                self.send_header(k, v)
            self.end_headers()
            return content

    # Uses urllib to parse the path of the current request.
    def translate_path(self):
        return urllib.parse.urlparse(self.path)

    # Override this to change the content produced by the handler. Returns a tuple of:
    #   response : the http response code, such as HTTPStatus.OK
    #   msg      : the message to send at the end of the http response line (or None for a default message)
    #   headers  : a list of tuples to send as MIME headers: (keyword, value)
    #              NOTE: do not put Content-Length in here, it is generated automatically in send_head
    #              NOTE: however, do put Content-Type in here if you want to send it!
    #   content  : the bytes you want to send as the body of this response
    def construct_content(self):
        pr = self.translate_path()

        assert pr.path != '/testing'

        response = HTTPStatus.NOT_FOUND
        msg = None
        headers = (
            ('Content-Type', 'text/html',),
        )
        body = (
            '<!DOCTYPE html>\n'
            '<html>\n'
            '  <head>\n'
            '    <title>404 Nothing Here</title>\n'
            '  </head>\n'
            '  <body>\n'
            '    <h1>There is no content on this server!</h1>\n'
            '  </body>\n'
            '</html>\n'
        )

        return response, msg, headers, bytes(body, encoding='ascii')

class AsyncTCPServer(ThreadingMixIn, TCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost',
                        help='server host')
    parser.add_argument('--port', type=int, default=8000,
                        help='server port')
    args = parser.parse_args()

    with AsyncTCPServer((args.host, args.port,), AsyncHTTPRequestHandler) as server:
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        print('Server on thread: {}.'.format(server_thread.name))
        print('Close stdin to stop.')

        for line in sys.stdin:
            pass

        print('Closed stdin, stopping...')
        server.shutdown()
        print('Goodbye!')
