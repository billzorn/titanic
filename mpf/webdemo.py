#!/usr/bin/env python

import describefloat
import webcontent
from aserver import AsyncCache, AsyncTCPServer, AsyncHTTPRequestHandler

class TitanicHTTPRequestHandler(AsyncHTTPRequestHandler):
    
    # subclass and overwrite these before using
    fmt_cache = None
    demo_cache = None
    the_pool = None

    # configuration
    limit_exp = 200000

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
            assert 2 <= w <= 20
        except Exception:
            w = webcontent.default_w

        try:
            p = int(args['p'])
            assert 2 <= p < 1024
        except Exception:
            p = webcontent.default_p

        try:
            s = str(args['s'])[:1536]
        except Exception:
            s = webcontent.default_s

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
            w, p, s = self.process_args(raw_args)
            show_format = path in {'fmt', 'jfmt'}
            the_pool = type(self).the_pool

            if show_format:
                the_cache = type(self).fmt_cache
                kargs = (w, p,)
            else:
                the_cache = type(self).demo_cache
                discrete, parsed = the_pool.apply(describefloat.parse_input,
                                                  (s, w, p, type(self).limit_exp, False,))
                if discrete:
                    S, E, T = parsed
                    kargs = (S.uint, E.uint, T.uint, w, p)
                else:
                    R = parsed
                    kargs = (str(R), w, p,)

            # get cached content
            cached = the_cache.lookup(kargs)
            if cached is None:
                hit = False
                if show_format:
                    cached = the_pool.apply(describefloat.process_format,
                                            (w, p,))
                else:
                    cached = the_pool.apply(describefloat.process_parsed_input,
                                            (s, w, p, discrete, parsed,))
                the_cache.update(kargs, cached)
            else:
                hit = True

            # construct page
            if show_format:
                content = cached
            else:
                content = describefloat.explain_input(s, w, p, discrete, parsed, hit=hit) + '\n\n' + cached
                if discrete:
                    w = E.n
                    p = T.n + 1

            body = (webcontent.skeleton_indent + '{}\n\n<br>').format(webcontent.create_webform(w, p, s))
            body.replace('\n', '\n' + webcontent.skeleton_indent)
            body += '\n\n' + webcontent.pre(content)

            return webcontent.skeletonize(body), 'text/html'
