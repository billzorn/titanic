#!/usr/bin/env python

# Quick and dirty web demo of symbolic floating point.

from http.server import BaseHTTPRequestHandler, HTTPStatus
import urllib
import socketserver

version = '0.0'

HOST = 'localhost'
PORT = 8000

max_w = 20
max_p = 1024
max_chars = 1000

default_w = '8'
default_p = '24'
default_chars = 'PI'

page = '''<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">    
    <title>SymFP</title>
  </head>

  <body>
    <header>
      <h1>SymFP</h1>
      <aside>Symbolic (infinite precision!) floating point</aside>
    </header>

    <br>

    <form action="/demo" method="get">
      w: <input type="number" name="w" value="{{}}" min="2" max="{}">
      p: <input type="number" name="p" value="{{}}" min="2" max="{}">
      <input type="text" name="s" value="{{}}" maxlength="{}">
      <input type="submit" value="Submit">
    </form>

    <br>

    <p>BTW you got path: {{}}</p>

  </body>
</html>
'''.format(str(max_w), str(max_p), str(max_chars))

def ghetto_arguments(s):
    fields = s.split('&')
    args = {}
    for f in fields:
        name_arg = f.split('=')
        if len(name_arg) == 2 and len(name_arg[0]) > 0:
            args[name_arg[0].strip()] = urllib.parse.unquote_plus(name_arg[1].strip())
    return args

class DemoHTTPRequestHandler(BaseHTTPRequestHandler):

    server_version = 'SymfpDemoHTTP/' + version

    def do_GET(self):
        f = self.send_head()
        if f:
            self.wfile.write(f.encode('utf-8'))
            self.wfile.write('\n'.encode('utf-8'))
            self.wfile.flush()

    def do_HEAD(self):
        self.send_head()

    # also constructs the html to send
    def send_head(self):
        args = self.translate_path(self.path)
        print(repr(args))
        if args is None:
            # homepage
            f = page.format(
                default_w,
                default_p,
                default_chars,
                repr(self.path)
            )
        else:
            # actually run a query
            f = page.format(
                args.get('w', default_w),
                args.get('p', default_p),
                args.get('s', default_chars),
                repr(self.path)
            )   
        ctype = 'text/html'
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-type", ctype)
            self.send_header("Content-Length", len(f))
            self.end_headers()
            return f
        except:
            # could do cleanup like closing a file
            raise

    def translate_path(self, path):
        pr = urllib.parse.urlparse(path)
        if pr.path == '/demo':
            return ghetto_arguments(pr.query)
        else:
            return None

Handler = DemoHTTPRequestHandler

with socketserver.TCPServer((HOST, PORT), Handler) as httpd:
    print('serving at port', PORT)
    httpd.serve_forever()
