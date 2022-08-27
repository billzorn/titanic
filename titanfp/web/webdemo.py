import sys
import os
import io
import threading
import traceback
import html
import json
import base64
import http
import multiprocessing

from PIL import Image
import numpy as np

# from .fserver import AsyncTCPServer, AsyncHTTPRequestHandler
from . import fserver

from ..titanic.utils import *

from ..fpbench import fpcparser, fpyparser, fpcast as ast
from ..titanic import utils, digital, ndarray
from ..arithmetic import interpreter, analysis
from ..arithmetic import ieee754, posit
#from ..arithmetic import softfloat, softposit
from ..arithmetic import sinking, sinkingposit
from ..arithmetic import interval
from ..arithmetic import mpmf

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
    'ieee754': ieee754.Interpreter,
    'posit': posit.Interpreter,
    #'softfloat': softfloat.Interpreter,
    #'softposit': softposit.Interpreter,
    'sinking-point': sinking.Interpreter,
    'sinking-posit': sinkingposit.Interpreter,
    'interval': interval.Interpreter,
    'mpmf': mpmf.Interpreter,
}

webdemo_float_backends = {
    'ieee754', 'softfloat', 'sinking-point'
}

webdemo_posit_backends = {
    'posit', 'softposit', 'sinking-posit'
}


webdemo_mpmf_backends = {
    'mpmf'
}

class WebtoolError(Exception):
    """Unable to run webtool; malformed data or bad options."""

class WebtoolParseError(WebtoolError):
    """Unable to parse FPCore."""

class WebtoolArgumentError(WebtoolError):
    """Unable to decipher FPCore arguments."""


class WebtoolState(object):

    def _read_int(self, x, name, minimum=None, maximum=None):
        try:
            i = int(x)
        except ValueError as e:
            raise WebtoolError(name + ' must be an integer')

        if minimum is not None and i < minimum:
            raise WebtoolError(name + ' must be at least ' + str(minimum))
        elif maximum is not None and i > maximum:
            raise WebtoolError(name + ' can be at most ' + str(maximum))
        else:
            return i

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
    img = None
    img_tensor = None
    enable_analysis = None
    heatmap = None

    def __init__(self, data):
        try:
            payload = json.loads(data.decode('utf-8'))
        except json.decoder.JSONDecodeError:
            raise WebtoolError('malformed json request data')

        if 'core' in payload:
            buf = str(payload['core']).strip()
            try:
                if buf.startswith('FPCore'):
                    self.cores = fpyparser.compile(buf)
                else:
                    self.cores = fpcparser.compile(buf)
            except fpcparser.FPCoreParserError as e:
                raise WebtoolParseError(str(e))
            except Exception:
                raise WebtoolParseError(traceback.format_exc())

        if 'args' in payload:
            try:
                self.args = fpcparser.read_exprs(str(payload['args']).strip())
            except fpcparser.FPCoreParserError as e:
                raise WebtoolArgumentError('invalid arguments for FPCore: ' + str(e))

        if 'backend' in payload:
            self.backend = str(payload['backend']).strip()

        if 'w' in payload:
            self.w = self._read_int(payload['w'], 'w', minimum=2, maximum=16)

        if 'p' in payload:
            self.p = self._read_int(payload['p'], 'p', minimum=2, maximum=1024)

        if 'float_override' in payload:
            self.float_override = self._read_bool(payload['float_override'], 'float_override')

        if 'es' in payload:
            self.es = self._read_int(payload['es'], 'es', minimum=0, maximum=16)

        if 'nbits' in payload:
            self.nbits = self._read_int(payload['nbits'], 'nbits', minimum=2, maximum=128)

        if 'posit_override' in payload:
            self.posit_override = self._read_bool(payload['posit_override'], 'posit_override')

        if 'usr_img' in payload:
            imgdata = payload['usr_img']

            try:
                decoded = base64.decodebytes(bytes(imgdata, 'ascii'))
                self.img = Image.open(io.BytesIO(decoded))
                self.img_tensor = np_array_to_ndarray(np.array(self.img))
            except Exception:
                print('Exception decoding user image:', file=sys.stderr, flush=True)
                traceback.print_exc()
                print('', file=sys.stderr, flush=True)

        if 'enable_analysis' in payload:
            self.enable_analysis = self._read_bool(payload['enable_analysis'], 'enable_analysis')

        if 'heatmap' in payload:
            self.heatmap = self._read_bool(payload['heatmap'], 'heatmap')

        self.payload = payload

    @property
    def precision(self):
        if self.backend in webdemo_float_backends:
            return ast.Data((ast.Var('float'), ast.Integer(self.w), ast.Integer(self.p + self.w)))
        elif self.backend in webdemo_posit_backends:
            return ast.Data((ast.Var('posit'), ast.Integer(self.es), ast.Integer(self.nbits)))
        else:
            return None

    @property
    def override(self):
        if self.backend in webdemo_float_backends:
            return self.float_override
        elif self.backend in webdemo_posit_backends:
            return self.posit_override
        elif self.backend in webdemo_mpmf_backends:
            return False
        else:
            return None


def np_array_to_sexp(a):
    if isinstance(a, np.ndarray):
        return '(' + ' '.join((np_array_to_sexp(elt) for elt in a)) + ')'
    else:
        return repr(a)

def img_to_sexp(img):
    a = np.array(img)
    return np_array_to_sexp(a)

# convert everything into integers, should be fine for images
def np_array_to_ndarray(a):
    data, shape = ndarray.reshape(a)
    return ndarray.NDArray(data=map(int, data), shape=shape)

def pixel(x):
    return max(0, min(int(x), 255))

def b64_encode_image(e):
    bitmap_tensor = ndarray.NDArray(shape=e.shape, data=map(pixel, e.data))
    bitmap = np.array(bitmap_tensor, dtype=np.uint8)
    img = Image.fromarray(bitmap)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    data = str(base64.encodebytes(buf.getvalue()), 'ascii')
    buf.close()
    return data


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def mkplot_b64(result_array):
    results = [[], [], []]
    for x,y,z in result_array:
        r_x, r_y, r_z = results
        r_x.append(float(str(x)))
        r_y.append(float(str(y)))
        r_z.append(float(str(z)))

    fig = plt.figure(figsize=(8,6), dpi=80)
    ax = Axes3D(fig)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.plot(results[0], results[1], results[2], color='blue', lw=1)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    data = str(base64.encodebytes(buf.getvalue()), 'ascii')
    plt.close(fig)
    buf.close()
    return data


def create_analysis_report(interpreter):
    report = 'Evaluated {:d} expressions\n\n'.format(interpreter.evals)
    reports = [als.report() for als in interpreter.analyses]
    return report + '\n\n'.join(reports)


def run_eval(data):
    #print('Eval yo!')
    #print(repr(data))

    try:

        state = WebtoolState(data)
        if state.backend in webdemo_eval_backends:
            backend = webdemo_eval_backends[state.backend]
            backend_interpreter = backend()
            #backend_interpreter.max_evals = 1000000
        else:
            raise WebtoolError('unknown Titanic evaluator backend: {}'.format(repr(state.backend)))
        if state.cores is None or len(state.cores) < 1:
            raise WebtoolError('must provide one or more FPCores')

        main = None
        for core in state.cores:
            backend_interpreter.register_function(core)
            if core.ident and core.ident.lower() == 'main':
                main = core
        if main is None:
            main = state.cores[-1]
        core = main

        nargs = len(state.args)
        extra_arg_msg = ''
        if state.img is not None:
            nargs += 1
            extra_arg_msg = '\n  If an image is provided, it will be passed in a tensor as the first argument to the core'
        if nargs != len(core.inputs):
            raise WebtoolArgumentError('expected {:d} arguments for FPCore, got {:d}:\n  {}{}'
                                       .format(len(core.inputs), len(state.args), ', '.join(map(str, state.args)), extra_arg_msg))

        props = {}
        precision = state.precision
        if precision is not None:
            props['precision'] = precision
        ctx = backend.ctype(props=props)

        print(ctx)

        # hack?
        if state.backend in webdemo_mpmf_backends:
            ctx = None # use context from the FPCore

        if state.img is not None:
            args_with_image = [state.img_tensor] + state.args
        else:
            args_with_image = state.args

        try:
            arg_ctx = backend_interpreter.arg_ctx(core, args_with_image, ctx=ctx, override=state.override)
            named_args = [[str(k), ('[' + 'x'.join(['{}'] * len(shape)) + ' tensor]').format(*arg_ctx.bindings[k].shape) if shape
                           else str(arg_ctx.bindings[k])]
                          for k, props, shape in core.inputs]

            # yuck
            if state.img is not None:
                rows, cols, channels = state.img_tensor.shape
                named_args[0][1] = '[{}x{} image]'.format(rows, cols)

            # reset the interpreter to avoid counting evals from arguments
            backend_interpreter = backend()

            if state.enable_analysis:
                #backend_interpreter.max_evals = 1000000
                als, bc_als, r_als = analysis.DefaultAnalysis(), analysis.BitcostAnalysis(), analysis.RangeAnalysis()
                backend_interpreter.analyses = [als, bc_als]
            else:
                pass
                #backend_interpreter.max_evals = 5000000

            for core in state.cores:
                backend_interpreter.register_function(core)

            try:
                pre_val = backend_interpreter.interpret_pre(core, args_with_image, ctx=ctx, override=state.override)
            except interpreter.EvaluatorError as e:
                pre_val = str(e)

            e_val = backend_interpreter.interpret(core, args_with_image, ctx=ctx, override=state.override)

            if state.enable_analysis:
                analysis_report = create_analysis_report(backend_interpreter)
            else:
                analysis_report = ''

        except interpreter.EvaluatorUnboundError as e:
            raise WebtoolError('unbound variable {}'.format(str(e)))
        except interpreter.EvaluatorError as e:
            raise WebtoolError(str(e))
        except utils.TitanicAbort:
            raise WebtoolError('interpreter aborted')

        result = {
            'success': 1,
            'args': named_args,
            'e_val': str(e_val),
            'pre_val': str(pre_val),
            'eval_count': str(backend_interpreter.evals),
        }

        if analysis_report:
            result['report'] = analysis_report

        made_plot = False

        if state.img is not None:
            if isinstance(e_val, ndarray.NDArray) and len(e_val.shape) == 3 and e_val.shape[2] in [3,4]:
                e_img = e_val
                # if state.heatmap:
                #     e_img = ndarray.NDArray(shape=e_img.shape, data=[
                #         (max(0, d.ctx.p - d.p) / d.ctx.p) * 255 for d in e_img.data
                #     ])
                result['result_img'] = b64_encode_image(e_img)
                result['e_val'] = 'image'
        elif state.heatmap:
            print('hm')
            if isinstance(e_val, ndarray.NDArray) and len(e_val.shape) == 2 and e_val.shape[1] == 3:
                print('  plotting')
                try:
                    result['result_img'] = mkplot_b64(e_val)
                    result['e_val'] = '3d plot (disable plot option to print array)'
                    made_plot = True
                except Exception:
                    made_plot = False
                    traceback.print_exc()

        # 2d matrix printer
        if (not made_plot) and isinstance(e_val, ndarray.NDArray) and len(e_val.shape) == 2:
            print(' -- 2d override')
            ndstr = ndarray.NDArray(shape=e_val.shape, data=[str(d) for d in e_val.data])
            result['mat_2d'] = ndstr.tolist()
            rows, cols = ndstr.shape
            result['e_val'] = '{} by {} matrix:'.format(rows, cols)

    except WebtoolError as e:
        traceback.print_exc()
        result = {
            'success': 0,
            'message' : str(e),
        }
    except:
        print('Exception during titanic evaluation:', file=sys.stderr, flush=True)
        traceback.print_exc()
        print('', file=sys.stderr, flush=True)
        result = {
            'success': 0,
            'message': 'internal evaluator error',
        }

    try:
        return json.dumps(result).encode('utf-8')
    except:
        print('Exception encoding titanic evaluator result:', file=sys.stderr, flush=True)
        traceback.print_exc()
        print('', file=sys.stderr, flush=True)
        return b'{"success": 0}'


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
