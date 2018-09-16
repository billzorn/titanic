import random
import math
import itertools
import multiprocessing

import numpy as np

from .fpbench import fpcparser
from .arithmetic import ieee754, sinking
from .arithmetic import core2math
from .arithmetic import evalctx
from .titanic import digital

from .titanic import gmpmath
from .titanic import wolfmath


def gen_input(e, p, nbits, negative=False):
    # p <= nbits

    significand = random.randint(0, (1 << (nbits - 1)) - 1) | (1 << (nbits - 1))

    hi = digital.Digital(negative=negative, c=significand, exp=e - nbits + 1)
    lo = hi.round_m(p)

    return hi, lo

def gen_e(e, c, negative=False):
    exp = e - c.bit_length() + 1
    return digital.Digital(negative=negative, exp=exp, c=c)


def linear_ulps(x, y):
    smaller_n = min(x.n, y.n)
    x_offset = x.n - smaller_n
    y_offset = y.n - smaller_n

    x_c = x.c << x_offset
    y_c = y.c << y_offset

    return x_c - y_c

def bits_agreement(hi, lo):
    bitsim = gmpmath.geo_sim(hi, lo)
    if math.isinf(bitsim):
        if bitsim > 0:
            agreement = max(hi.p, lo.p)
        else:
            agreement = 1
    else:
        agreement = int(bitsim) + 4

    hi_exact = digital.Digital(hi, inexact=False, rc=0)
    lo_exact = digital.Digital(lo, inexact=False, rc=0)
    one_ulp_agreement = None
    zero_ulp_agreement = None
    for p in range(agreement, -1, -1):
        hi_rounded = hi_exact.round_m(p)
        lo_rounded = lo_exact.round_m(p)
        rounded_ulps = linear_ulps(hi_rounded, lo_rounded)
        if one_ulp_agreement is None and abs(rounded_ulps) <= 1:
            one_ulp_agreement = p
        if zero_ulp_agreement is None and rounded_ulps == 0:
            zero_ulp_agreement = p

        if one_ulp_agreement is not None and zero_ulp_agreement is not None:
            break

    if one_ulp_agreement == None:
        one_ulp_agreement = 0
    if zero_ulp_agreement == None:
        zero_ulp_agreement = 0

    # if agreement > 0 and (agreement <= one_ulp_agreement or agreement <= zero_ulp_agreement):
    #     print('possibly underestimated agreement:\n  {} vs {}\n  {}, {}, {}, {}'
    #           .format(hi, lo, bitsim, agreement, one_ulp_agreement, zero_ulp_agreement))

    return bitsim, one_ulp_agreement, zero_ulp_agreement

ctx4096 = evalctx.IEEECtx(w=32, p=4096)
ctx128 = evalctx.IEEECtx(w=32, p=128)
ctx64 = evalctx.IEEECtx(w=16, p=64)
ctx32 = evalctx.IEEECtx(w=16, p=32)
ctx_double = evalctx.IEEECtx(w=11, p=53)
ctx512 = evalctx.IEEECtx(w=20, p=512)

rejections = 100
progress_update = 10000
batchsize = 20000

def gen_core_arguments(core, es, ps, nbits, ctx):
    hi_args, lo_args = zip(*[gen_input(es[i], ps[i], nbits) for i in range(len(core.inputs))])

    if core.pre is not None:
        for reject in range(rejections):
            if (ieee754.Interpreter.interpret_pre(core, hi_args, ctx) and
                ieee754.Interpreter.interpret_pre(core, lo_args, ctx)):
                break
            hi_args, lo_args = zip(*[gen_input(es[i], ps[i], nbits) for i in range(len(core.inputs))])

        if not (ieee754.Interpreter.interpret_pre(core, hi_args, ctx) and
                ieee754.Interpreter.interpret_pre(core, lo_args, ctx)):
            raise ValueError('failed to meet precondition of fpcore:\n{}'
                             .format(str(core)))

    return hi_args, lo_args


def bench_core(core, hi_args, lo_args, ctx):
    hi_result = ieee754.Interpreter.interpret(core, hi_args, ctx=ctx)
    lo_result = ieee754.Interpreter.interpret(core, lo_args, ctx=ctx)
    sunk = sinking.Interpreter.interpret(core, lo_args, ctx)

    if sunk.inexact:
        p = sunk.p
    else:
        p = float('inf')

    return [p, *bits_agreement(hi_result, lo_result)]


def iter_1arg(erange, prange, benches):
    for e1 in erange:
        for p1 in prange:
            for i in range(benches):
                yield [e1], [p1]

def iter_2arg(erange, prange, benches):
    for e1 in erange:
        for e2 in erange:
            for p1 in prange:
                for p2 in prange:
                    for i in range(benches):
                        yield [e1, e2], [p1, p2]


def sweep(core, cases, nbits, ctx):
    records = []

    for es, ps in cases:
        try:
            hi_args, lo_args = gen_core_arguments(core, es, ps, nbits, ctx)
        except Exception as e:
            if progress_update > 0:
                print('!', end='', flush=True)
            continue

        records.append(bench_core(core, hi_args, lo_args, ctx))
        if progress_update > 0 and len(records) % progress_update == 0:
            print('.', end='', flush=True)

    return records


def sweep_single(core, cases, nbits, ctx):
    print('{:s}\nrunning with {:d} total bits'.format(str(core), nbits), flush=True)

    records = sweep(core, cases, nbits, ctx)

    if progress_update > 0:
        print('\ngenerated {:d} records'.format(len(records)), flush=True)
    else:
        print('generated {:d} records'.format(len(records)), flush=True)

    return records


# break arguments up into chunks manually.
# thanks to:
# https://stackoverflow.com/questions/8991506/iterate-an-iterator-by-chunks-of-n-in-python
def grouper(n, iterable):
    it = iter(iterable)
    while True:
       chunk = tuple(itertools.islice(it, n))
       if not chunk:
           return
       yield chunk

def sweep_multi(core, cases, nbits, ctx, nprocs=None):
    if nprocs is None:
        nprocs = max(multiprocessing.cpu_count() // 2, 1)

    print('{:s}\nrunning with {:d} total bits on {:d} processes'.format(str(core), nbits, nprocs), flush=True)

    pool = multiprocessing.Pool(processes=nprocs)
    arg_iter = ([core, chunk, nbits, ctx] for chunk in grouper(batchsize, cases))

    all_records = []
    map_invocations = 0
    for records in pool.starmap(sweep, arg_iter):
        map_invocations += 1
        if records is not None:
            all_records += records

    if progress_update > 0:
        print('\nstarmap finished with {:d} invocations'.format(map_invocations), flush=True)
    else:
        print('starmap finished with {:d} invocations'.format(map_invocations), flush=True)

    pool.close()
    pool.join()

    print('generated {:d} records'.format(len(all_records)), flush=True)

    return all_records


benchmarks = {
    'nop' : '(FPCore (x) x)',

    'add' : '(FPCore (x y) (+ x y))',
    'sub' : '(FPCore (x y) (- x y))',
    'mul' : '(FPCore (x y) (* x y))',
    'div' : '(FPCore (x y) (/ x y))',
    'sqrt' : '(FPCore (x) (sqrt x))',

    # 'floor' : '(FPCore (x) (floor x))',
    # 'fmod' : '(FPCore (x y) (fmod x y))',
    # 'sin' : '(FPCore (x) (sin x))',
    # 'pow' : '(FPCore (x y) (pow x y))',

    'quadratic' : """(FPCore (a b c)
 :name "NMSE p42, positive"
 :cite (hamming-1987 herbie-2015)
 :fpbench-domain textbook
 :pre (and (>= (* b b) (* 4 (* a c))) (!= a 0))
 (/ (+ (- b) (sqrt (- (* b b) (* 4 (* a c))))) (* 2 a)))
""",

    'ex3.1' : """(FPCore (x)
 :name "NMSE example 3.1"
 :cite (hamming-1987 herbie-2015)
 :fpbench-domain textbook
 :pre (>= x 0)
 (- (sqrt (+ x 1)) (sqrt x)))
""",

    'ex3.6' : """(FPCore (x)
 :name "NMSE example 3.6"
 :cite (hamming-1987 herbie-2015)
 :fpbench-domain textbook
 :pre (>= x 0)
 (- (/ 1 (sqrt x)) (/ 1 (sqrt (+ x 1)))))
""",

    'ex3.3.1' : """(FPCore
 (x)
 :name "NMSE problem 3.3.1"
 :cite (hamming-1987 herbie-2015)
 :fpbench-domain textbook
 :pre (!= x 0)
 (- (/ 1 (+ x 1)) (/ 1 x)))
""",

    'ex3.3.3' : """(FPCore (x)
 :name "NMSE problem 3.3.3"
 :cite (hamming-1987 herbie-2015)
 :fpbench-domain textbook
 :pre (!= x 0 1 -1)
 (+ (- (/ 1 (+ x 1)) (/ 2 x)) (/ 1 (- x 1))))
""",

    'herbified_quadratic': """(FPCore (a b c)
 :herbie-status success
 :herbie-time 118637.2451171875
 :herbie-bits-used 3392
 :herbie-error-input ((256 28.609971950677362) (8000 33.90307227594979))
 :herbie-error-output ((256 5.078369297841056) (8000 6.594164753178634))
 :name "NMSE p42, positive"
 :pre (and (>= (* b b) (* 4 (* a c))) (!= a 0))
 (if (<= b -3.2964251401560902e+93)
     (- (/ b a))
     (if (<= b -9.121837495335558e-234)
         (/ 1 (/ (* a 2) (- (sqrt (- (* b b) (* c (* a 4)))) b)))
         (if (<= b 4.358108025323294e+96)
             (/ 1 (* (+ (sqrt (- (* b b) (* c (* a 4)))) b) (/ 2 (/ (- 4) (/ 1 c)))))
             (/ (- c) b)))))
""",

    'herbified_ex3.1': """(FPCore (x)
 :herbie-status success
 :herbie-time 24932.283935546875
 :herbie-bits-used 1344
 :herbie-error-input ((256 30.095582124185807) (8000 29.789458520339096))
 :herbie-error-output ((256 0.18978500976844204) (8000 0.16283740625180287))
 :name "NMSE example 3.1"
 :pre (>= x 0)
 (/ 1 (+ (sqrt (+ x 1)) (sqrt x))))
""",

    'herbified_ex3.6': """(FPCore (x)
 :herbie-status success
 :herbie-time 26448.68994140625
 :herbie-bits-used 1088
 :herbie-error-input ((256 17.200122070308325) (8000 19.44276561824272))
 :herbie-error-output ((256 0.38451646722105215) (8000 0.43027832341504624))
 :name "NMSE example 3.6"
 :pre (>= x 0)
 (* (/ (sqrt (/ 1 (+ (sqrt (+ x 1)) (sqrt x)))) (sqrt x)) (/ (sqrt (/ 1 (+ (sqrt (+ x 1)) (sqrt x)))) (sqrt (+ x 1)))))
""",

    'herbified_ex3.3.1': """(FPCore (x)
 :herbie-status success
 :herbie-time 19409.0048828125
 :herbie-bits-used 832
 :herbie-error-input ((256 12.87691845436457) (8000 14.037291740926072))
 :herbie-error-output ((256 0.078125) (8000 0.0705))
 :name "NMSE problem 3.3.1"
 :pre (!= x 0)
 (/ (/ (- 1) (+ x 1)) x))
""",

    'herbified_ex3.3.3': """(FPCore (x)
 :herbie-status success
 :herbie-time 110668.86499023438
 :herbie-bits-used 1088
 :herbie-error-input ((256 9.663075481047713) (8000 9.645467709556803))
 :herbie-error-output ((256 0.06640625) (8000 0.07328308281331129))
 :name "NMSE problem 3.3.3"
 :pre (!= x 0 1 -1)
 (/ (/ 2 (* (+ x 1) x)) (- x 1)))
""",
}

cores = { k : fpcparser.compile1(v) for k, v in benchmarks.items() }
#maths = { k : core2math.compile(v) for k, v in cores.items() }



def split_records(records, xidx, yidx):
    xs = []
    ys = []
    for record in records:
        xs.append(record[xidx])
        ys.append(record[yidx])
    return xs, ys

def cdf(xs):
    mass = len(xs)
    xs_sorted = sorted(xs)
    ys = [i / mass for i in range(len(xs))]
    return xs_sorted, ys

def split_xs(xs, ys):
    cdfs = {}
    for x, y in zip(xs, ys):
        if x not in cdfs:
            cdfs[x] = []
        cdfs[x].append(y)
    return cdfs


# from ipython
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def do_scatter(xs, ys, title, fname):
    fig, ax = plt.subplots()

    fig.set_size_inches(8, 5.5)

    ax.set_xlim(0, 12)
    ax.set_ylim(-3, 15)
    ax.set_xlabel('sinking-point reported precision (bits)')
    ax.set_ylabel('actual precision (bits accuracy)')
    ax.set_title(title)

    ax.scatter(xs, ys, alpha=0.002)

    xlim = ax.get_xlim()
    blackline = mlines.Line2D(xlim, xlim, color='black')
    redline = mlines.Line2D(xlim, [y-1 for y in xlim], color='red')
    ax.add_line(blackline)
    ax.add_line(redline)

    fig.savefig(fname)

def do_cdf(xs, ys, title, fname, use_line2=False):
    fig, ax = plt.subplots()

    fig.set_size_inches(8, 5.5)

    ax.set_xlim(-4, 12)
    ax.set_ylim(0.0, 1.1)
    ax.set_xlabel('excess precision (bits accuracy)')
    ax.set_ylabel('fraction of results')
    ax.set_title(title)

    #ax.set_xlim(-2, 2)
    #ax.set_ylim(-0.1, 0.3)

    cdfs = split_xs(xs, ys)

    for x, ys2 in cdfs.items():
        cdf_xs, cdf_ys = cdf([y - x for y in ys2])
        ax.plot(cdf_xs, cdf_ys)

    x_min, x_max = ax.get_xlim()
    ref_x = np.linspace(x_min, x_max, 1000)
    ref_y = [1 - (1) * (2**-x) for x in ref_x]
    ref_y2 = [1 - (0.5) * (2**-x) for x in ref_x]

    line = mlines.Line2D(ref_x, ref_y, color='black', linestyle='--', linewidth=1)
    line2 = mlines.Line2D(ref_x, ref_y2, color='black', linestyle='-.', linewidth=1)
    ax.add_line(line)
    if use_line2:
        ax.add_line(line2)

    ax.axvline(x=0.0, color='black', linewidth=1)
    fig.savefig(fname)

def make_figs():
    import matplotlib
    matplotlib.rcParams.update({'font.size': 16})

    erange = range(-14, 16)
    prange = range(1, 12)
    nbits = 32
    ctx = ctx64

    for corename in ['nop', 'add', 'sub', 'mul', 'div', 'sqrt']:

        core = cores[corename]

        if len(core.inputs) == 1:
            argiter = iter_1arg(erange, prange, 3000)
        else:
            argiter = iter_2arg(erange, prange, 10)

        if corename == 'sub':
            line2 = True
        else:
            line2 = False

        data = sweep_multi(core, argiter, nbits, ctx)
        xs, ys = split_records(data, 0, 1)

        do_scatter(xs, ys, 'accuracy for ' + corename, 'fig/' + corename + '_scatter.png')
        do_cdf(xs, ys, 'excess precision for ' + corename, 'fig/' + corename + '_cdf.png', line2)





# some random testing for larger programs
import numpy
import sys

_r_f64_max = (1 << 64) - 1
_r_f32_max = (1 << 32) - 1

def random_float64():
    bits = random.randint(1, _r_f64_max)
    return float(numpy.frombuffer(bits.to_bytes(8, sys.byteorder),dtype=numpy.float64, count=1, offset=0))

def random_float32():
    bits = random.randint(1, _r_f32_max)
    return float(numpy.frombuffer(bits.to_bytes(4, sys.byteorder),dtype=numpy.float32, count=1, offset=0))


corenames = [
    'quadratic',
    'ex3.1',
    'ex3.6',
    'ex3.3.1',
    'ex3.3.3'
]
corenames += ['herbified_' + name for name in corenames]




def gen_random_double_arguments(core):
    rargs = [random_float64() for arg in core.inputs]

    while not ieee754.Interpreter.interpret_pre(core, rargs, ctx=ctx512):
        rargs = [random_float64() for arg in core.inputs]

    return rargs


def run_example(corename, n):
    core = cores[corename]

    total_sink_prec = 0.0
    total_bits_acc = 0.0
    total_sink_acc = 0.0

    rejects = 0

    for trial in range(n):
        rargs = gen_random_double_arguments(core)

        exactish_result = ieee754.Interpreter.interpret(core, rargs, ctx=ctx512)
        sinking_result = sinking.Interpreter.interpret(core, rargs, ctx=ctx_double)

        bits_acc = gmpmath.geo_sim(exactish_result, sinking_result)

        if math.isnan(bits_acc):
            rejects += 1
        else:

            bits_acc = min(max(0, bits_acc), 53)

            total_sink_prec += sinking_result.p
            total_bits_acc += bits_acc
            total_sink_acc += (bits_acc - sinking_result.p)

    print('{}: {:d} trials with {:d} rejections'.format(corename, n, rejects))
    successes = n - rejects
    return total_sink_prec / successes, total_bits_acc / successes, total_sink_acc / successes

def make_table():
    for name in corenames:
        sink_prec, bits_acc, sink_acc = run_example(name, 10000)
        print(repr(sink_prec), repr(bits_acc), repr(sink_acc))
