import sys
import os
import subprocess
import random
import math

import numpy
import sfpy

from .arithmetic import evalctx
from .arithmetic.canonicalize import Canonicalizer, Condenser
from .arithmetic import native, np
from .arithmetic import softfloat, softposit
from .arithmetic import ieee754, posit
from .fpbench import fpcparser, fpcast


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

def random_float(nbits):
    if nbits == 64:
        return float(sfpy.Float64(random.randint(0, 0xffffffffffffffff)))
    elif nbits == 32:
        return float(sfpy.Float32(random.randint(0, 0xffffffff)))
    elif nbits == 16:
        return float(sfpy.Float16(random.randint(0, 0xffff)))
    else:
        raise ValueError('nbits must be 64, 32, or 16, got: {}'.format(nbits))

def random_posit(nbits):
    if nbits == 32:
        return float(sfpy.Posit32(random.randint(0, 0xffffffff)))
    if nbits == 16:
        return float(sfpy.Posit16(random.randint(0, 0xffff)))
    if nbits == 8:
        return float(sfpy.Posit8(random.randint(0, 0xff)))
    else:
        raise ValueError('nbits must be 32, 16, or 8, got: {}'.format(nbits))

def type_to_precision(cores):
    for core in cores:
        if 'type' in core.props and 'precision' not in core.props:
            core.props['precision'] = core.props.pop('type')

def strip_precision(cores):
    for core in cores:
        if 'type' in core.props or 'precision' in core.props:
            core.props = core.props.copy()
            if 'type' in core.props:
                core.props.pop('type')
            if 'precision' in core.props:
                core.props.pop('precision')


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
            args.append(random_float(64))
        elif prec in evalctx.binary32_synonyms:
            args.append(random_float(32))
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
            args.append(random_float(64))
            run_posit = False
        elif prec in evalctx.binary32_synonyms:
            args.append(random_float(32))
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

fctxs = [
    evalctx.IEEECtx(props={'precision': fpcast.Var('binary64')}),
    evalctx.IEEECtx(props={'precision': fpcast.Var('binary32')}),
    evalctx.IEEECtx(props={'precision': fpcast.Var('binary16')}),
]

def test_float(core, ctx):
    args = [random_float(ctx.w + ctx.p) for name, props in core.inputs]
    sf_answer = float(softfloat.Interpreter.interpret(core, args, ctx=ctx))
    ieee_answer = float(ieee754.Interpreter.interpret(core, args, ctx=ctx))

    isexact = (math.isnan(sf_answer) and math.isnan(ieee_answer)) or sf_answer == ieee_answer
    isclose = isexact or math.isclose(sf_answer, ieee_answer)

    if not isclose:
        print('failure on {}\n  {}\n  native={} vs. np={}'.format(
            str(core.name), repr(args), repr(sf_answer), repr(ieee_answer),
        ))
    elif not isexact:
        print('mismatch on {}\n  {}\n  native={} vs. np={}'.format(
            str(core.name), repr(args), repr(sf_answer), repr(ieee_answer)
        ))

    return not isclose

pctxs = [
    evalctx.PositCtx(props={'precision': fpcast.Var('binary32')}),
    evalctx.PositCtx(props={'precision': fpcast.Var('binary16')}),
    evalctx.PositCtx(props={'precision': fpcast.Var('binary8')}),
]

def test_posit(core, ctx):
    args = [random_posit(ctx.nbits) for name, props in core.inputs]
    sp_answer = float(softposit.Interpreter.interpret(core, args, ctx=ctx))
    posit_answer = float(posit.Interpreter.interpret(core, args, ctx=ctx))

    isexact = (((math.isinf(sp_answer) or math.isnan(sp_answer))
                and (math.isinf(posit_answer) or math.isnan(posit_answer)))
               or sp_answer == posit_answer)
    isclose = isexact or math.isclose(sp_answer, float(posit_answer))

    if not isclose:
        print('failure on {}\n  {}\n  native={} vs. np={}'.format(
            str(core.name), repr(args), repr(sp_answer), repr(posit_answer),
        ))
    elif not isexact:
        print('mismatch on {}\n  {}\n  native={} vs. np={}'.format(
            str(core.name), repr(args), repr(sp_answer), repr(posit_answer)
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






def run_test(test, cores, reps=10, ctx=None):
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
                if ctx is None:
                    attempt = test(core)
                else:
                    attempt = test(core, ctx)
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


def test_posit_conversion(es, nbits):
    ptype = softposit.softposit_precs[(es, nbits)]
    posit_values = [float(ptype(i)) for i in range(1 << nbits)]
    posit_values.sort()

    nearby_cases = set()
    for a in posit_values:
        nearby_cases.add(float(numpy.nextafter(a, -numpy.inf)))
        nearby_cases.add(float(numpy.nextafter(a, numpy.inf)))

    arithmetic_means = set()
    geometric_means = set()
    for a, b in zip(posit_values, posit_values[1:]):
        mean = (a + b) / 2
        arithmetic_means.add(float(mean))
        nearby_cases.add(float(numpy.nextafter(mean, -numpy.inf)))
        nearby_cases.add(float(numpy.nextafter(mean, numpy.inf)))

        geomean = math.sqrt(a * b)
        geometric_means.add(float(geomean))
        nearby_cases.add(float(numpy.nextafter(geomean, -numpy.inf)))
        nearby_cases.add(float(numpy.nextafter(geomean, numpy.inf)))

    cases = set().union(posit_values, arithmetic_means, geometric_means, nearby_cases)
    more_cases = set()

    for case in cases:
        more_cases.add(case)
        more_cases.add(-case)
        if case == 0.0:
            more_cases.add(float('inf'))
            more_cases.add(float('-inf'))
        else:
            more_cases.add(1/case)
            more_cases.add(-1/case)

    sorted_cases = sorted(more_cases)

    print('{:d} test cases for rounding'.format(len(sorted_cases)))

    for f in sorted_cases:
        softposit_answer = ptype(f)
        posit_answer = posit.Posit(f, ctx=posit.posit_ctx(es, nbits))

        if not float(softposit_answer) == float(posit_answer):
            print('case {}: {} != {}'.format(repr(f), str(softposit_answer), str(posit_answer)))



def rounding_cases(dtype, nbits, maxcases=None):
    if maxcases is None:
        values = [float(dtype(i)) for i in range(1 << nbits)]
    else:
        imax = (1 << nbits) - 1
        values = set()
        for case in range(maxcases):
            i = random.randint(0, imax)
            if i > 0:
                values.add(float(dtype(i-1)))
            values.add(float(dtype(i)))
            if i < imax:
                values.add(float(dtype(i+1)))

    values = sorted(values)

    nearby_values = set()
    for a in values:
        nearby_values.add(float(numpy.nextafter(a, -numpy.inf)))
        nearby_values.add(float(numpy.nextafter(a, numpy.inf)))

    arithmetic_means = set()
    geometric_means = set()
    for a, b in zip(values, values[1:]):
        mean = (a + b) / 2
        arithmetic_means.add(float(mean))
        nearby_values.add(float(numpy.nextafter(mean, -numpy.inf)))
        nearby_values.add(float(numpy.nextafter(mean, numpy.inf)))

        try:
            geomean = math.sqrt(a * b)
            geometric_means.add(float(geomean))
            nearby_values.add(float(numpy.nextafter(geomean, -numpy.inf)))
            nearby_values.add(float(numpy.nextafter(geomean, numpy.inf)))
        except Exception:
            pass

    cases = set().union(values, arithmetic_means, geometric_means, nearby_values)
    more_cases = set()

    for case in cases:
        if not math.isnan(case):
            more_cases.add(case)
            more_cases.add(-case)
            if case == 0.0:
                more_cases.add(float('inf'))
                more_cases.add(float('-inf'))
            else:
                more_cases.add(1/case)
                more_cases.add(-1/case)

    return sorted(more_cases)

def test_posit_rounding(es, nbits, maxcases=None):
    dtype = softposit.softposit_precs[(es, nbits)]
    ctx = posit.posit_ctx(es, nbits)
    cases = rounding_cases(dtype, nbits, maxcases=maxcases)

    print('Testing posit rounding on {:d} cases...'.format(len(cases)), flush=True)
    for f in cases:
        softposit_answer = dtype(f)
        posit_answer = posit.Posit(f, ctx=ctx)
        if not (float(softposit_answer) == float(posit_answer)):
            print('  case {}: {} != {}'.format(repr(f), str(softposit_answer), str(posit_answer)))
    print('... Done.', flush=True)

def test_float_rounding(w, p, maxcases=None):
    dtype = softfloat.softfloat_precs[(w, p)]
    ctx = ieee754.ieee_ctx(w, p)
    cases = rounding_cases(dtype, w+p, maxcases=maxcases)

    print('Testing float rounding on {:d} cases...'.format(len(cases)), flush=True)
    for f in cases:
        softfloat_answer = dtype(f)
        ieee754_answer = ieee754.Float(f, ctx=ctx)
        if not (float(softfloat_answer) == float(ieee754_answer)):
            print('  case {}: {} != {}'.format(repr(f), str(softfloat_answer), str(ieee754_answer)))
    print('... Done.', flush=True)



test_posit_rounding(1, 16)
test_float_rounding(5, 11)

## KNOWN ISSUES WITH OLD ROUNDING METHOD ############################

# Testing posit rounding on 1112637 cases...
#   case -inf: inf != nan
#   case -25165824.0: -16777216.0 != -3.4e+07
#   case -2.2351741790771484e-08: -1.4901161193847656e-08 != -3e-08
#   case -7.450580596923828e-09: -1.4901161193847656e-08 != -3.7e-09
#   case 7.450580596923828e-09: 1.4901161193847656e-08 != 3.7e-09
#   case 2.2351741790771484e-08: 1.4901161193847656e-08 != 3e-08
#   case 25165824.0: 16777216.0 != 3.4e+07
#   case inf: inf != nan
# ... Done.
# Testing float rounding on 1077869 cases...
# ... Done.

# real    0m46.941s
# user    0m47.057s
# sys     0m0.400s

#####################################################################





# if __name__ == '__main__':
#     smalltext = filter_cores('operators', '+', '-', '*', '/', 'sqrt', 'nearbyint',
#                              '<', '<=', '>', '>=', '==', '!=', 'fmin', 'fmax',
#                              'isfinite', 'isinf', 'isnan', 'isnormal', 'signbit')
#     smallcores = fpcparser.compile(smalltext)
#     type_to_precision(smallcores)

#     alltext = filter_cores('name')
#     allcores = fpcparser.compile(alltext)
#     type_to_precision(allcores)

#     print(len(smallcores), len(allcores))

#     barecores = fpcparser.compile(smalltext)
#     strip_precision(barecores)

#     def go():
#         #run_test(test_canon, allcores, reps=1)
#         #run_test(test_native_np, allcores, reps=10)

#         for ctx in fctxs:
#             run_test(test_float, barecores, reps=50, ctx=ctx)
#         for ctx in pctxs:
#             run_test(test_posit, barecores, reps=50, ctx=ctx)
