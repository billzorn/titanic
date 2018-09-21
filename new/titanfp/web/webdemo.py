#!/usr/bin/env python

import sys
import os
import threading
import traceback
import json
import multiprocessing
import subprocess
import http
import html
import argparse

from .aserver import AsyncCache, AsyncTCPServer, AsyncHTTPRequestHandler

from ..fpbench import fpcparser
from ..arithmetic import native, np
from ..arithmetic import softfloat, softposit
from ..arithmetic import ieee754, posit
from ..arithmetic import canonicalize
from ..arithmetic import evalctx


here = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(here, 'index.html'), 'rb') as f:
    index = f.read()
with open(os.path.join(here, 'titanic.css'), 'rb') as f:
    css = f.read()
with open(os.path.join(here, 'titanfp.min.js'), 'rb') as f:
    bundle = f.read()

with open(os.path.join(here, '../../../www/favicon.ico'), 'rb') as f:
    favicon = f.read()
with open(os.path.join(here, '../../../www/piceberg_round.png'), 'rb') as f:
    logo = f.read()

fpbench_root = '/home/bill/private/research/origin-FPBench'
fpbench_tools = os.path.join(fpbench_root, 'tools')
fpbench_benchmarks = os.path.join(fpbench_root, 'benchmarks')

def run_tool(toolname, core, *args):
    tool = subprocess.Popen(
        args=['racket', os.path.join(fpbench_tools, toolname), *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_data, stderr_data = tool.communicate(input=core.sexp.encode('utf-8'))

    success = True
    retval = tool.wait()
    if retval != 0:
        success = False
        print('subprocess:\n  {}\nreturned {:d}'.format(' '.join(tool.args), retval),
              file=sys.stderr, flush=True)

    if stderr_data:
        print(stderr_data, file=sys.stderr, flush=True)

    return success, stdout_data.decode('utf-8')

def filter_cores(*args, benchmark_dir = fpbench_benchmarks):
    if not os.path.isdir(benchmark_dir):
        raise ValueError('{}: not a directory'.format(benchmark_dir))

    names = os.listdir(benchmark_dir)
    benchmark_files = [name for name in names
                       if name.lower().endswith('.fpcore')
                       and os.path.isfile(os.path.join(benchmark_dir, name))]

    cat = subprocess.Popen(
        cwd=benchmark_dir,
        args=['cat', *benchmark_files],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    cat.stdin.close()

    tool = subprocess.Popen(
        args=['racket', os.path.join(fpbench_tools, 'filter.rkt'), *args],
        stdin=cat.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_data, stderr_data = tool.communicate()

    # cleanup
    success = True
    for proc in [cat, tool]:
        retval = proc.wait()
        if retval != 0:
            success = False
            print('subprocess:\n  {}\nreturned {:d}'.format(' '.join(proc.args), retval),
                  file=sys.stderr, flush=True)

    cat_stderr_data = cat.stderr.read()
    cat.stderr.close()
    if cat_stderr_data:
        print(cat_stderr_data, file=sys.stderr, flush=True)

    if stderr_data:
        print(stderr_data, file=sys.stderr, flush=True)

    return success, stdout_data.decode('utf-8')


def demo_tool(success, output):
    if success:
        return output
    else:
        return 'Error - tool subprocess returned nonzero value'

def demo_arith(evaluator, arguments, core, ctx=None):
    if arguments is None:
        try:
            return str(evaluator(core))
        except Exception:
            print('Exception in FPCore evaluation\n  evaluator={}\n  args={}\n  core={}'
                  .format(repr(evaluator), repr(arguments), core.sexp))
            traceback.print_exc()
            return 'Error evaluating FPCore.'
    else:
        inputs = arguments.strip().split()
        if len(inputs) != len(core.inputs):
            return 'Error - wrong number of arguments (core expects {:d})'.format(len(core.inputs))
        try:
            return str(evaluator(core, inputs, ctx))
        except Exception:
            print('Exception in FPCore evaluation\n  evaluator={}\n  args={}\n  core={}'
                  .format(repr(evaluator), repr(arguments), core.sexp))
            traceback.print_exc()
            return 'Error evaluating FPCore.'

class RaisingArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError('unable to parse inputs')


DEFAULT_PROPAGATE = {'precision', 'round', 'math-library'}
DEFAULT_RECURSE = {'pre', 'spec'}
def parse_canon_args(args):
    parser = RaisingArgumentParser(add_help=False)
    parser.add_argument('--default', action='store_true')
    parser.add_argument('--recurse', type=str, nargs='*')
    parser.add_argument('--propagate', type=str, nargs='*')
    ns = parser.parse_args(args.strip().split())
    if ns.recurse is None and ns.propagate is None:
        return DEFAULT_RECURSE, DEFAULT_PROPAGATE
    if ns.recurse is None:
        recurse = set()
    else:
        recurse = set(ns.recurse)
    if ns.propagate is None:
        propagate = set()
    else:
        propagate = set(ns.propagate)
    if ns.default:
        recurse.update(DEFAULT_RECURSE)
        propagate.update(DEFAULT_PROPAGATE)
    return recurse, propagate

def demo_canon(evaluator, arguments, core, use_prop=False):
    try:
        recurse, propagate = parse_canon_args(arguments)
    except Exception:
        print('Exception parsing arguments for canonicalizer: {}'.format(repr(arguments)))
        traceback.print_exc()
        return 'Error parsing arguments.'
    try:
        if use_prop:
            return evaluator(core, recurse=recurse, propagate=propagate).sexp
        else:
            return evaluator(core, recurse=recurse).sexp
    except Exception:
        print('Exception in FPCore translation\n  translator={}\n  recurse={}\n  propagate={}\n  use_prop={}\n  core={}'
              .format(repr(evaluator), repr(recurse), repr(propagate), repr(use_prop), core.sexp))
        traceback.print_exc()
        return 'Error translating FPCore.'


class TitanfpHTTPRequestHandler(AsyncHTTPRequestHandler):

    def construct_content(self, data):
        pr = self.translate_path()

        if pr.path == '/titanfp.min.js':
            response = http.server.HTTPStatus.OK
            msg = None
            headers = (
                ('Content-Type', 'text/javascript'),
            )
            content = bundle

        elif pr.path == '/titanic.css':
            response = http.server.HTTPStatus.OK
            msg = None
            headers = (
                ('Content-Type', 'text/css'),
            )
            content = css
        
        else:
            response = http.server.HTTPStatus.OK
            msg = None

            if data is None:
                if pr.path == '/favicon.ico':
                    headers = (
                        ('Content-Type', 'image/x-icon'),
                    )
                    content = favicon
                elif pr.path == '/piceberg_round.png':
                    headers = (
                        ('Content-Type', 'image/png'),
                    )
                    content = logo
                else:
                    headers = (
                        ('Content-Type', 'text/html'),
                    )
                    content = index
            else:
                try:
                    payload = json.loads(data.decode('utf-8'))
                except Exception as e:
                    print('Malformed data payload:\n{}'.format(repr(data)))
                    traceback.print_exc()

                try:
                    core = fpcparser.compile(payload['core'])[0]
                except Exception:
                    print('Exception parsing FPCore {}'.format(repr(payload['core'])))
                    traceback.print_exc()
                    core = None
                    output = 'Error - unable to parse FPCore'

                try:
                    if core is not None:
                        backend = payload['backend']
                        if backend == 'ieee754':
                            ctx = ieee754.ieee_ctx(int(payload['w']), int(payload['p']))
                            output = demo_arith(ieee754.Interpreter.interpret, payload['inputs'], core, ctx)
                        elif backend == 'posit':
                            ctx = posit.posit_ctx(int(payload['es']), int(payload['nbits']))
                            output = demo_arith(posit.Interpreter.interpret, payload['inputs'], core, ctx)
                        elif backend == 'native':
                            output = demo_arith(native.Interpreter.interpret, payload['inputs'], core)
                        elif backend == 'np':
                            output = demo_arith(np.Interpreter.interpret, payload['inputs'], core)
                        elif backend == 'softfloat':
                            output = demo_arith(softfloat.Interpreter.interpret, payload['inputs'], core)
                        elif backend == 'softposit':
                            output = demo_arith(softposit.Interpreter.interpret, payload['inputs'], core)
                        elif backend == 'canonicalize':
                            output = demo_canon(canonicalize.Canonicalizer.translate, payload['inputs'], core, use_prop=True)
                        elif backend == 'condense':
                            output = demo_canon(canonicalize.Condenser.translate, payload['inputs'], core, use_prop=False)
                        elif backend == 'minimize':
                            output = demo_canon(canonicalize.Minimizer.translate, payload['inputs'], core, use_prop=False)
                        elif backend == 'fpcore':
                            inputs = payload['inputs'].strip().split()
                            if len(inputs) != len(core.inputs):
                                output = 'Error - wrong number of arguments (core expects {:d})'.format(len(core.inputs))
                            else:
                                output = demo_tool(*run_tool('fpcore.rkt', core, *inputs))
                        elif backend == 'core2c':
                            output = demo_tool(*run_tool('core2c.rkt', core))
                        elif backend == 'core2js':
                            output = demo_tool(*run_tool('core2js.rkt', core))
                        elif backend == 'core2smtlib2':
                            output = demo_tool(*run_tool('core2smtlib2.rkt', core))
                        # elif backend == 'filter':
                        #     inputs = payload['inputs'].strip().split()
                        #     output = demo_tool(*filter_cores(*inputs))
                        else:
                            output = 'Unknown backend ' + repr(backend)

                except Exception as e:
                    print('Exception running backend\n  payload={}'.format(repr(payload)))
                    traceback.print_exc()
                    output = 'Error running backend.'

                headers = (
                    ('Content-Type', 'text/plain'),
                )
                content = html.escape(str(output)).encode('utf-8')

        return response, msg, headers, content


def run():
    import argparse

    ncores = os.cpu_count()
    #default_pool_size = max(1, min(ncores - 1, (ncores // 2) + 1))
    default_pool_size = 2

    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=int, default=1,
                        help='number of requests to cache')
    parser.add_argument('--workers', type=int, default=default_pool_size,
                        help='number of worker processes to run in parallel')
    parser.add_argument('--host', type=str, default='localhost',
                        help='server host')
    parser.add_argument('--port', type=int, default=8000,
                        help='server port')
    args = parser.parse_args()

    cache = AsyncCache(args.cache)
    with multiprocessing.Pool(args.workers, maxtasksperchild=100) as pool:
        class CustomHTTPRequestHandler(TitanfpHTTPRequestHandler):
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


if __name__ == '__main__':
    run()
