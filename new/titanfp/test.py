import sys
import os
import subprocess
import random
import math

import numpy

from .arithmetic import evalctx
from .arithmetic.canonicalize import Canonicalizer, Condenser
from .arithmetic import native, np
from .arithmetic import softfloat, softposit
from .arithmetic import ieee754
from .fpbench import fpcparser


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

    retval = tool.wait()
    if retval != 0:
        print('subprocess:\n  {}\nreturned {:d}'.format(' '.join(tool.args), retval),
              file=sys.stderr, flush=True)

    if stderr_data:
        print(stderr_data, file=sys.stderr, flush=True)

    return stdout_data.decode('utf-8')

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
    for proc in [cat, tool]:
        retval = proc.wait()
        if retval != 0:
            print('subprocess:\n  {}\nreturned {:d}'.format(' '.join(proc.args), retval),
                  file=sys.stderr, flush=True)

    cat_stderr_data = cat.stderr.read()
    cat.stderr.close()
    if cat_stderr_data:
        print(cat_stderr_data, file=sys.stderr, flush=True)

    if stderr_data:
        print(stderr_data, file=sys.stderr, flush=True)

    return stdout_data.decode('utf-8')


_r_f64_max = (1 << 64) - 1
_r_f32_max = (1 << 32) - 1

def random_float64():
    bits = random.randint(1, _r_f64_max)
    return float(numpy.frombuffer(bits.to_bytes(8, sys.byteorder),dtype=numpy.float64, count=1, offset=0))

def random_float32():
    bits = random.randint(1, _r_f32_max)
    return float(numpy.frombuffer(bits.to_bytes(4, sys.byteorder),dtype=numpy.float32, count=1, offset=0))


def type_to_precision(cores):
    for core in cores:
        if 'type' in core.props and 'precision' not in core.props:
            core.props['precision'] = core.props.pop('type')


def test_canon(core):
    ref = run_tool('canonicalizer.rkt', core)
    try:
        ref_core = fpcparser.compile1(ref)
    except ValueError:
        print('could not parse output:\n{}'.format(ref))
        return True

    ti_canon = Canonicalizer.translate(core)
    ti_cond = Condenser.translate(core)

    ti_canon2 = Canonicalizer.translate(ti_cond)
    ti_cond2 = Condenser.translate(ti_canon)

    failed = False
    if not ti_canon == ti_canon2:
        print('canonicalization failure on {}!'.format(str(core.name)))
        print('canon       = {}'.format(ti_canon.sexp))
        print('canon(cond) = {}'.format(ti_canon2.sexp))
        failed = True
    if not ti_cond == ti_cond2:
        print('condensation failed on {}!'.format(str(core.name)))
        print('cond        = {}'.format(ti_cond.sexp))
        print('cond(canon) = {}'.format(ti_cond2.sexp))
        failed = True

    if not ti_canon.e == ref_core.e:
        print('canonicalization failed vs FPBench on {}!'.format(str(core.name)))
        print('canon   = {}'.format(ti_canon.sexp))
        print('FPBench = {}'.format(ref_core.sexp))
        failed = True

    return failed

def test_native_np(core):
    ctx = evalctx.EvalCtx(props=core.props)
    args = []
    run_native = True
    for name, props in core.inputs:
        argctx = ctx.let(props=props)
        prec = str(argctx.props.get('precision', 'binary64')).strip().lower()
        if prec in evalctx.binary64_synonyms:
            args.append(random_float64())
        elif prec in evalctx.binary32_synonyms:
            args.append(random_float32())
            run_native = False
        else:
            return None

    if run_native:
        native_answer = native.Interpreter.interpret(core, args)
    else:
        native_answer = None

    np_answer = np.Interpreter.interpret(core, args)

    if native_answer is None:
        return None

    else:
        isexact = (math.isnan(native_answer) and math.isnan(float(np_answer))) or native_answer == np_answer
        isclose = isexact or math.isclose(native_answer, float(np_answer))

        if not isclose:
            print('failure on {}\n  {}\n  native={} vs. np={}'.format(
                str(core.name), repr(args), repr(native_answer), repr(np_answer),
            ))
        elif not isexact:
            print('mismatch on {}\n  {}\n  native={} vs. np={}'.format(
                str(core.name), repr(args), repr(native_answer), repr(np_answer)
            ))

        return not isclose

def test_np_softfloat_ieee754(core):
    npctx = evalctx.EvalCtx(props=core.props)
    sfctx = evalctx.IEEECtx(props=core.props)
    spctx = evalctx.IEEECtx(props=core.props)
    ieeectx = evalctx.IEEECtx(props=core.props)
    args = []
    run_posit = True
    for name, props in core.inputs:
        np_argctx = npctx.let(props=props)
        prec = str(np_argctx.props.get('precision', 'binary64')).strip().lower()
        if prec in evalctx.binary64_synonyms:
            args.append(random_float64())
            run_posit = False
        elif prec in evalctx.binary32_synonyms:
            args.append(random_float32())
        else:
            return None

    np_answer = float(np.Interpreter.interpret(core, args))
    sf_answer = float(softfloat.Interpreter.interpret(core, args))
    ieee_answer = float(ieee754.Interpreter.interpret(core, args))

    isexact = (math.isnan(sf_answer) and math.isnan(np_answer) and math.isnan(ieee_answer)) or sf_answer == np_answer == ieee_answer
    isclose = isexact or (math.isclose(sf_answer, np_answer) and math.isclose(ieee_answer, np_answer))

    if not isclose:
        print('failure on {}\n  {}\n  np={} vs. sf={} vs. ieee={}'.format(
            str(core.name), repr(args), repr(np_answer), repr(sf_answer), repr(ieee_answer)
        ))
    elif not isexact:
        print('mismatch on {}\n  {}\n  np={} vs. sf={} vs. ieee={}'.format(
            str(core.name), repr(args), repr(np_answer), repr(sf_answer), repr(ieee_answer)
        ))

    if run_posit:
        sp_answer = float(softposit.Interpreter.interpret(core, args))
        sp_isclose = math.isinf(sf_answer) or math.isnan(sf_answer) or math.isclose(sf_answer, sp_answer, rel_tol=1e-01)
        if not sp_isclose:
            print('posit mismatch on {}\n  {}\n  sp={} vs. sf={}'.format(
                str(core.name), repr(args), repr(sp_answer), repr(sf_answer)
            ))

    return not isclose






setup = """
class A(object):
    foo = 0

n_flat_classes = 1000
n_child_classes = 1000

flat_classes = []
for i in range(n_flat_classes):
    name = 'B_' + str(i)
    flat_classes.append(type(name, (A,), dict(foo=i + 1)))

_last_child_class = A
child_classes = []
for i in range(n_child_classes):
    name = 'C_' + str(i)
    _last_child_class = type(name, (_last_child_class,), dict(foo=i + 1 + n_child_classes))
    child_classes.append(_last_child_class)

B0 = flat_classes[0]()
B69 = flat_classes[69]()
A0 = child_classes[0]()
A69 = child_classes[69]()
A_1 = child_classes[-1]()

flat_dispatch = {cls:cls.foo for cls in flat_classes}
child_dispatch = {cls:cls.foo for cls in child_classes}
"""

import timeit






def run_test(test, cores, reps=10):
    print('Running test {} on {:d} cores...'.format(repr(test), len(cores)))
    i = 0
    attempts = 0
    failures = 0
    for core in cores:
        try:
            print('cores[{:d}] {} '.format(i, str(core.name)), end='', flush=True)
            any_attempts = False
            any_fails = False
            for rep in range(reps):
                attempt = test(core)
                any_attempts = any_attempts or (attempt is not None)
                any_fails = any_fails or attempt
                if attempt:
                    print('!', end='', flush=True)
                else:
                    print('.', end='', flush=True)

            print('')
            if any_attempts:
                attempts += 1
            if any_fails:
                failures += 1

        except KeyboardInterrupt:
            print('ABORT', flush=True)
            continue
        finally:
            i += 1
    print('\n...Done. {:d} attempts, {:d} failures.'.format(attempts, failures))


if __name__ == '__main__':
    smalltext = filter_cores('operators', '+', '-', '*', '/', 'sqrt', 'nearbyint',
                             '<', '<=', '>', '>=', '==', '!=', 'fmin', 'fmax',
                             'isfinite', 'isinf', 'isnan', 'isnormal', 'signbit')
    smallcores = fpcparser.compile(smalltext)
    type_to_precision(smallcores)

    alltext = filter_cores('name')
    allcores = fpcparser.compile(alltext)
    type_to_precision(allcores)

    print(len(smallcores), len(allcores))

    def go():
        #run_test(test_canon, allcores, reps=1)
        #run_test(test_native_np, allcores, reps=10)
        run_test(test_np_softfloat_ieee754, smallcores, reps=1000)
