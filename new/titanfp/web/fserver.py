# aserver: parallel webserver in one file

import sys
import os
import threading
import traceback
import html
import urllib
import json
import mimetypes
from multiprocessing import Pool
from http.server import BaseHTTPRequestHandler, HTTPStatus
from socketserver import ThreadingMixIn, TCPServer


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


# quick demo
WEBPAGE_MESSAGE = '''\
<!DOCTYPE html>
<html>
  <head>
    <title>{title:s}</title>
  </head>
  <body>
<pre>
{body:s}
<pre>
  </body>
</html>
'''

WEBPAGE_CONTENT_TYPE = 'text/html'

def test_worker_parse(qs):
    # diagnostic info
    ppid = os.getppid()
    pid = os.getpid()
    active_threads = threading.active_count()
    current_thread = threading.current_thread()
    # do some work
    query_list = urllib.parse.parse_qsl(qs)
    return (ppid, pid, active_threads, current_thread.name,), query_list


class AsyncHTTPRequestHandler(BaseHTTPRequestHandler):

    # configuration
    server_version = 'fserver/0.5'
    sys_version = "Python/" + sys.version.split()[0]
    protocol_version = 'HTTP/1.0'
    error_message_format = ERROR_MESSAGE.format(server_version, sys_version)
    error_content_type = ERROR_CONTENT_TYPE
    traceback_log = True
    traceback_send = True

    # subclass and override to process requests in parallel
    the_pool = None

    # async calls
    def apply(self, fn, args):
        if self.the_pool is None:
            return fn(*args)
        else:
            return self.the_pool.apply(fn, args)

    # These do not need to be overridden again; they just call send_head.
    def do_HEAD(self):
        self.send_head()

    def do_GET(self):
        content = self.send_head(None)
        if content is not None:
            try:
                self.wfile.write(content)
            except Exception as e:
                self.log_error('Caught {} while writing response to GET.\n\n{}',
                               repr(e), traceback.format_exc())

    def do_POST(self):
        try:
            length = int(self.headers['Content-Length'])
            data = self.rfile.read(length)
        except Exception as e:
            self.log_error('Caught {} while reading post data.\n\n{}',
                               repr(e), traceback.format_exc())
            data = False
        content = self.send_head(data)
        if content is not None:
            try:
                self.wfile.write(content)
            except Exception as e:
                self.log_error('Caught {} while writing response to POST.\n\n{}',
                               repr(e), traceback.format_exc())

    # This does not need to be overridden again: it just calls self.construct_content().
    # Override that in subclasses to change the content produced by the handler.
    def send_head(self, data):
        try:
            response, msg, headers, content = self.construct_content(data)

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

    # avoid sending unescaped strings that might break the console
    def log_request(self, code='-', size='-'):
        if isinstance(code, HTTPStatus):
            code = code.value
        self.log_message('%s %s', repr(self.requestline), str(code))

    def log_message(self, format, *args):
        sys.stderr.write("%s [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format%args))

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
    def construct_content(self, data):
        pr = self.translate_path()

        assert pr.path != '/crash'

        if pr.path.startswith('/test'):
            # diagnostic info
            ppid = os.getppid()
            pid = os.getpid()
            active_threads = threading.active_count()
            current_thread = threading.current_thread()
            handler_diag = ppid, pid, active_threads, current_thread.name

            # parse the query (a)synchronously
            worker_diag, parsed = self.apply(test_worker_parse, (pr.query,))

            # spit it all out
            req_format = 'raw request:\n{}\n\ncommand: {}\npath:    {}\nversion: {}\n\nheaders:\n{}'
            diag_format = '{}\n  ppid: {:d}\n  pid: {:d}\n  active threads: {:d}\n  current thread: {}'
            body_text = ('{}\n\n{}\n\n{}\n\npath: {}\nquery: {}'
                         .format(
                             req_format.format(self.raw_requestline, self.command, self.path, self.request_version, self.headers),
                             diag_format.format('Handler:', *handler_diag),
                             diag_format.format('Parser worker:', *worker_diag),
                             repr(pr.path),
                             repr(parsed),
                         ))

            response = HTTPStatus.OK
            msg = None
            headers = (
                ('Content-Type', WEBPAGE_CONTENT_TYPE,),
            )
            body = WEBPAGE_MESSAGE.format(title=html.escape('aserver test', quote=False),
                                          body=html.escape(body_text, quote=False))

            return response, msg, headers, bytes(body, encoding='ascii')

        else:
            response = HTTPStatus.NOT_FOUND
            msg = None
            headers = (
                ('Content-Type', WEBPAGE_CONTENT_TYPE,),
            )
            body = WEBPAGE_MESSAGE.format(title=html.escape('404 Nothing Here', quote=False),
                                          body=html.escape('There is no content on this server!', quote=False))

            return response, msg, headers, bytes(body, encoding='ascii')

class AsyncTCPServer(ThreadingMixIn, TCPServer):
    allow_reuse_address = True
    daemon_threads = True


# quick and dirty, serve a directory
def serve_flat_directory(root):
    if not os.path.isdir(root):
        raise ValueError('FServer must serve a directory')

    print('Serving files from {}...'.format(root))

    content = {}
    content_bytes = 0
    for fname in os.listdir(root):
        fpath = os.path.join(root, fname)
        if os.path.isfile(fpath):
            with open(fpath, 'rb') as f:
                fcont = f.read()
            ftype = mimetypes.guess_type(fname)
            print('  {}, {}, {} bytes'.format(fname, str(ftype), len(fcont)))
            content[fname.lstrip('/')] = (ftype, fcont)
            content_bytes += len(fcont)

    print('Found {} files, {} bytes total.'.format(len(content), content_bytes))

    return content

class FServerRequestHandler(AsyncHTTPRequestHandler):

    # override this in a subclass to serve something
    the_content = {}

    def construct_content(self, data):
        pr = self.translate_path()
        path = pr.path.lstrip('/')

        if path in self.the_content:
            ctype_enc, cont = self.the_content[path]

            ctype, enc = ctype_enc

            response = HTTPStatus.OK
            msg = None
            headers = (
                ('Content-Type', ctype),
            )
            body = cont

        elif path == '' and 'index.html' in self.the_content:
            ctype, cont = self.the_content['index.html']

            response = HTTPStatus.OK
            msg = None
            headers = (
                ('Content-Type', ctype),
            )
            body = cont

        else:
            response = HTTPStatus.NOT_FOUND
            msg = None
            headers = (
                ('Content-Type', WEBPAGE_CONTENT_TYPE,),
            )
            body = bytes(
                WEBPAGE_MESSAGE.format(title=html.escape('404 Not Found', quote=False),
                                       body=html.escape('Nothing to see here.', quote=False)),
                encoding='ascii')


        return response, msg, headers, body


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost',
                        help='server host')
    parser.add_argument('--port', type=int, default=8000,
                        help='server port')
    parser.add_argument('--workers', type=int, default=2,
                        help='number of worker processes to run in parallel')
    parser.add_argument('--serve', type=str, default='',
                        help='serve a directory')
    args = parser.parse_args()

    with Pool(args.workers) as pool:


        print('{:d} worker processes.'.format(args.workers))

        if args.serve:
            class CustomHTTPRequestHandler(FServerRequestHandler):
                the_pool = pool
                the_content = serve_flat_directory(args.serve)

        else:
            class CustomHTTPRequestHandler(AsyncHTTPRequestHandler):
                the_pool = pool

        with AsyncTCPServer((args.host, args.port,), CustomHTTPRequestHandler) as server:
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
