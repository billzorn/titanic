#!/usr/bin/env python

import sys
import os
import threading
import html
import urllib
from multiprocessing import Pool
from http.server import HTTPStatus

import describefloat
import webcontent
from aserver import AsyncCache, AsyncTCPServer, AsyncHTTPRequestHandler

def get_first(d, k, default):
    x = d.get(k, (default,))
    return x[0]

# to distinguish different kinds of things in the cache
_FMT, _REAL, _DISCRETE = 0, 1, 2

class TitanicHTTPRequestHandler(AsyncHTTPRequestHandler):

    # configuration
    limit_exp = 200000
    traceback_log = True
    traceback_send = False

    def process_path(self):
        pr = self.translate_path()
        err, path, args = None, None, None

        pr_path = pr.path

        # empty path --> root page
        if webcontent.empty_re.fullmatch(pr_path):
            path = webcontent.root_page

        # static path
        if path is None:
            m = webcontent.page_re.fullmatch(pr_path)
            if m:
                path = m.group(1)

        # dynamic content by protocol
        if path is None:
            m = webcontent.protocol_re.fullmatch(pr_path)
            if m:
                path = m.group(1)
                args = urllib.parse.parse_qs(pr.query)

        # signal an error
        if path is None:
            err_title = 'Unknown Request'
            err_message = ('The requested path "{}" does not exist on the server.'
                           .format(html.escape(pr.path, quote=False)))

            response = HTTPStatus.NOT_FOUND
            msg = err_title
            headers = (
                ('Content-Type', 'text/html',),
            )
            body = webcontent.skeletonize(webcontent.create_error(err_title, err_message),
                                          ind=True)

            err = response, msg, headers, webcontent.webencode(body)

        return err, path, args

    def process_args(self, args):
        complaints = []
        err, s, w, p = None, None, None, None

        s = get_first(args, 's', webcontent.default_s)
        if len(s) > webcontent.maxlen:
            s = s[:webcontent.maxlen]
            complaints.append('Can only input at most {:d} characters.'.format(webcontent.maxlen))

        w = get_first(args, 'w', webcontent.default_w)
        if not isinstance(p, int):
            try:
                w = int(w)
            except Exception:
                complaints.append('w must be an integer.')
                w = webcontent.default_w
        if not (2 <= w <= webcontent.maxw):
            complaints.append('w must be between 2 and {:d}.'.format(webcontent.maxw))

        p = get_first(args, 'p', webcontent.default_p)
        if not isinstance(p, int):
            try:
                p = int(p)
            except Exception:
                complaints.append('p must be an integer.')
                p = webcontent.default_p
        if not (2 <= p <= webcontent.maxp):
            complaints.append('p must be between 2 and {:d}.'.format(webcontent.maxp))

        if complaints:
            err_complaints = 'bad input:\n  ' + '\n  '.join(complaints)

            response = HTTPStatus.OK
            msg = None
            headers, content = webcontent.protocol_headers_body(s, w, p, err_complaints)

            err = response, msg, headers, content

        return err, s, w, p

    def construct_content(self):
        err, path, args = self.process_path()
        if err is not None:
            return err

        # static page
        if args is None:
            data, ctype = webcontent.page_content[path]

            response = HTTPStatus.OK
            msg = None
            # could do something more clever, for caching etc.
            headers = (
                ('Content-Type', ctype,),
            )
            content = data

            return response, msg, headers, content

        err, s, w, p = self.process_args(args)
        if err is not None:
            return err

        # dynamic protocol
        protocol = path

        if protocol == 'fmt':
            prefix_results = False
            work_fn = describefloat.process_format
            work_args = w, p
            cache_key = _FMT, w, p
        elif protocol == 'demo':
            prefix_results = True
            discrete, parsed = self.apply(describefloat.parse_input,
                                          (s, w, p, self.limit_exp, False,))

            if discrete is None:
                err_complaints = 'bad input:\n  ' + parsed

                response = HTTPStatus.OK
                msg = None
                headers, content = webcontent.protocol_headers_body(s, w, p, err_complaints)

                return response, msg, headers, content

            work_fn = describefloat.process_parsed_input
            work_args = s, w, p, discrete, parsed
            if discrete:
                S, E, T = parsed
                cache_key = _DISCRETE, S.uint, E.uint, T.uint, w, p
            else:
                R = parsed
                cache_key = _REAL, repr(R), w, p
        else:
            raise ValueError('unknown protocol {}'.format(repr(protocol)))

        cache_hit, result = self.apply_cached(cache_key, work_fn, work_args)

        if prefix_results:
            prefix = describefloat.explain_input(s, w, p, discrete, parsed, hit=cache_hit) + '\n\n'
            if discrete:
                w = E.n
                p = T.n + 1
        else:
            prefix = ''

        response = HTTPStatus.OK
        msg = None
        headers, content = webcontent.protocol_headers_body(s, w, p, prefix + result)

        return response, msg, headers, content


if __name__ == '__main__':
    import argparse

    ncores = os.cpu_count()
    default_pool_size = max(1, min(ncores - 1, (ncores // 2) + 1))

    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=int, default=10000,
                        help='number of requests to cache')
    parser.add_argument('--workers', type=int, default=default_pool_size,
                        help='number of worker processes to run in parallel')
    parser.add_argument('--host', type=str, default='localhost',
                        help='server host')
    parser.add_argument('--port', type=int, default=8000,
                        help='server port')
    args = parser.parse_args()

    cache = AsyncCache(args.cache)
    with Pool(args.workers, maxtasksperchild=100) as pool:
        class CustomHTTPRequestHandler(TitanicHTTPRequestHandler):
            the_cache = cache
            the_pool = pool

        print('caching {:d} requests'.format(args.cache))
        print('{:d} worker processes'.format(args.workers))

        with AsyncTCPServer((args.host, args.port,), CustomHTTPRequestHandler) as server:
            server_thread = threading.Thread(target=server.serve_forever)
            server_thread.daemon = True
            server_thread.start()

            print('server on thread:', server_thread.name)
            print('close stdin to stop.')

            for line in sys.stdin:
                pass

            print('stdin closed, stopping.')
            pool.close()
            print('workers closing...')
            pool.join()
            print('workers joined successfully.')
            server.shutdown()
            print('goodbye!')
