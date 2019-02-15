import sys
import os
import threading
import traceback
import html
import json
import http
import multiprocessing

# from .fserver import AsyncTCPServer, AsyncHTTPRequestHandler
from . import fserver

from ..fpbench import fpcparser, fpcast as ast
from ..arithmetic import ieee754, posit
from ..arithmetic import softfloat, softposit


here = os.path.dirname(os.path.realpath(__file__))
dist = os.path.join(here, 'dist')

def path_aliases(path):
    stripped_path = path.lstrip('/')
    yield stripped_path
    if stripped_path == '':
        yield 'index.html'
    if not stripped_path.endswith('.html'):
        yield stripped_path + '.html'

webdemo_eval_backends = {
    'ieee754': ieee754.Interpreter.interpret,
    'posit': posit.Interpreter.interpret,
    'softfloat': softfloat.Interpreter.interpret,
    'softposit': softposit.Interpreter.interpret,
}

webdemo_float_backends = {
    'ieee754', 'softfloat',
}

webdemo_posit_backends = {
    'posit', 'softposit',
}


class WebtoolError(Exception):
    """Unable to run webtool; malformed data or bad options."""

class WebtoolParseError(WebtoolError):
    """Unable to parse FPCore."""

class WebtoolArgumentError(WebtoolError):
    """Unable to decipher FPCore arguments."""


class WebtoolState(object):

    def _read_int(self, x, name):
        try:
            return int(x)
        except ValueError as e:
            raise WebtoolError(name + ' must be an integer')

    def _read_bool(self, x, name):
        if x is True or x is False:
            return x
        else:
            s = str(x).lower()
            if s == 'true':
                return True
            elif s == 'false':
                return False
            else:
                raise WebtoolError(name + ' must be a boolean value')

    cores = None
    args = None
    backend = None
    w = 11
    p = 53
    float_override = False
    es = 4
    nbits = 64
    posit_override = False

    def __init__(self, data):
        try:
            payload = json.loads(data.decode('utf-8'))
        except json.decoder.JSONDecodeError:
            raise WebtoolError('malformed json request data')

        if 'core' in payload:
            try:
                self.cores = fpcparser.compile(payload['core'])
            except fpcparser.FPCoreParserError as e:
                raise WebtoolParseError(str(e))

        if 'args' in payload:
            try:
                self.args = fpcparser.read_exprs(payload['args'])
            except fpcparser.FPCoreParserError as e:
                raise WebtoolArgumentError('invalid arguments for FPCore: ' + str(e))

        if 'backend' in payload:
            self.backend = str(payload['backend']).strip()

        if 'w' in payload:
            self.w = self._read_int(payload['w'], 'w')

        if 'p' in payload:
            self.p = self._read_int(payload['p'], 'p')

        if 'float_override' in payload:
            self.float_override = self._read_bool(payload['float_override'], 'float_override')

        if 'es' in payload:
            self.es = self._read_int(payload['es'], 'es')

        if 'nbits' in payload:
            self.nbits = self._read_int(payload['nbits'], 'nbits')

        if 'posit_override' in payload:
            self.posit_override = self._read_bool(payload['posit_override'], 'posit_override')

        self.payload = payload

    @property
    def precision(self):
        if self.backend in webdemo_float_backends:
            return ast.Data([ast.Var('float'), ast.Decnum(str(self.w)), ast.Decnum(str(self.p))])
        elif self.backend in webdemo_posit_backends:
            return ast.Data([ast.Var('posit'), ast.Decnum(str(self.es)), ast.Decnum(str(self.nbits))])
        else:
            return None

def run_eval(data):
    try:
        state = WebtoolState(data)
        if state.cores is None or len(state.cores) != 1:
            raise WebtoolError('must provide exactly one FPCore')
        core = state.cores[0]
        if len(state.args) != len(core.inputs):
            raise WebtoolArgumentError('expected {:d} arguments for FPCore; got {:d}'
                                       .format(len(core.inputs), len(state.args)))

        if state.backend in webdemo_eval_backends:
            interpreter = webdemo_eval_backends[state.backend]
        else:
            raise WebtoolError('unknown Titanic evaluator backend: {}'.format(repr(state.backend)))


        result = {
            'core': core.sexp,
            'args': ' '.join(str(arg) for arg in state.args),
            'prec': str(state.precision),
        }

    except WebtoolError as e:
        result = {
            'error' : str(e),
        }
    except:
        print('Exception during titanic evaluation:', file=sys.stderr, flush=True)
        traceback.print_exc()
        print('', file=sys.stderr, flush=True)
        result = {
            'bad': '1',
        }

    try:
        return json.dumps(result).encode('utf-8')
    except:
        print('Exception encoding titanic evaluator result:', file=sys.stderr, flush=True)
        traceback.print_exc()
        print('', file=sys.stderr, flush=True)
        return b'{"bad": 1}'


class TitanicHTTPRequestHandler(fserver.AsyncHTTPRequestHandler):

    # create a subclass and override this to serve static content
    the_content = {}

    def construct_content(self, data):
        pr = self.translate_path()

        # static content
        for alias in path_aliases(pr.path):
            if alias in self.the_content:
                (ctype, enc), cont = self.the_content[alias]

                response = http.server.HTTPStatus.OK
                msg = None
                headers = (
                    ('Content-Type', ctype),
                )
                body = cont

                return response, msg, headers, body

        stripped_path = pr.path.lstrip('/')

        # dynamic content
        if stripped_path == 'eval':
            result = self.apply(run_eval, (data,))

            response = http.server.HTTPStatus.OK
            msg = None
            headers = (
                ('Content-Type', 'application/json'),
            )
            body = result

            return response, msg, headers, body

        # not found
        response = http.server.HTTPStatus.NOT_FOUND
        msg = None
        headers = (
            ('Content-Type', fserver.WEBPAGE_CONTENT_TYPE,),
        )
        body = bytes(
            fserver.WEBPAGE_MESSAGE.format(title=html.escape('404 Not Found', quote=False),
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

    with multiprocessing.Pool(args.workers) as pool:

        print('{:d} worker processes.'.format(args.workers))

        if args.serve:
            class CustomHTTPRequestHandler(TitanicHTTPRequestHandler):
                the_pool = pool
                the_content = fserver.serve_flat_directory(args.serve)

        else:
            class CustomHTTPRequestHandler(TitanicHTTPRequestHandler):
                the_pool = pool

        with fserver.AsyncTCPServer((args.host, args.port,), CustomHTTPRequestHandler) as server:
            server_thread = threading.Thread(target=server.serve_forever)
            server_thread.daemon = True
            server_thread.start()

            print('Titanic webdemo on thread: {}.'.format(server_thread.name))
            print('Close stdin to stop.')

            for line in sys.stdin:
                pass

            print('Closed stdin, stopping...')
            server.shutdown()
            print('Goodbye!')
