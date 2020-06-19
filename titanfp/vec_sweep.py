import math
import random
import operator
import traceback

from .titanic import ndarray
from .fpbench import fpcparser
from .arithmetic import mpmf, ieee754, posit, fixed, evalctx, analysis
from .arithmetic.mpmf import Interpreter

from .sweep import search
from .sweep.utils import *

dotprod_naive_template = '''(FPCore dotprod ((A n) (B m))
 :pre (== n m)
 {overall_prec}
  (for ([i n])
    ([accum 0 (! {sum_prec} (+ accum
                (! {mul_prec} (* (ref A i) (ref B i)))))])
    (cast accum)))
'''

dotprod_fused_template = '''(FPCore dotprod ((A n) (B m))
 :pre (== n m)
 {overall_prec}
  (for ([i n])
    ([accum 0 (! {sum_prec} (fma (ref A i) (ref B i) accum))])
    (cast accum)))
'''

dotprod_fused_unrounded_template = '''(FPCore dotprod ((A n) (B m))
 :pre (== n m)
 {overall_prec}
  (for ([i n])
    ([accum 0 (! {sum_prec} (fma (ref A i) (ref B i) accum))])
    accum))
'''


binsum_template = '''(FPCore addpairs ((A n))
 :pre (> n 1)
  (tensor ([i (# (/ (+ n 1) 2))])
    (let* ([k1 (# (* i 2))]
           [k2 (# (+ k1 1))])
      (if (< k2 n)
          (! {sum_prec} (+ (ref A k1) (ref A k2)))
          (ref A k1)))
  ))

(FPCore binsum ((A n))
  (while (> (size B 0) 1)
    ([B A (addpairs B)])
    (if (== (size B 0) 0) 0 (ref B 0))))
'''

nksum_template = '''(FPCore nksum ((A n))
 :name "Neumaier's improved Kahan Summation algorithm"
 {sum_prec}
  (for* ([i n])
    ([elt 0 (ref A i)]
     [t 0 (+ accum elt)]
     [c 0 (if (>= (fabs accum) (fabs elt))
              (+ c (+ (- accum t) elt))
              (+ c (+ (- elt t) accum)))]
     [accum 0 t])
    (+ accum c)))
'''

vec_prod_template = '''(FPCore vec-prod ((A n) (B m))
 :pre (== n m)
  (tensor ([i n])
    (! {mul_prec} (* (ref A i) (ref B i)))))
'''

dotprod_bin_template = (
    binsum_template + '\n' +
    vec_prod_template + '\n' +
'''(FPCore dotprod ((A n) (B m))
 :pre (== n m)
  (let ([result (binsum (vec-prod A B))])
    (! {overall_prec} (cast result))))
''')

dotprod_neumaier_template = (
    nksum_template + '\n' +
    vec_prod_template + '\n' +
'''(FPCore dotprod ((A n) (B m))
 :pre (== n m)
  (let ([result (nksum (vec-prod A B))])
    (! {overall_prec} (cast result))))
''')

def mk_dotprod(template, overall_prec, mul_prec, sum_prec):
    return template.format(overall_prec=overall_prec,
                           mul_prec=mul_prec,
                           sum_prec=sum_prec)


def largest_representable(ctx):
    if isinstance(ctx, evalctx.IEEECtx):
        return mpmf.MPMF(ctx.fbound, ctx)
    elif isinstance(ctx, evalctx.PositCtx):
        return mpmf.MPMF(m=1, exp=ctx.emax, ctx=ctx)
    else:
        raise ValueError(f'unsupported type: {type(ctx)!r}')

def smallest_representable(ctx):
    if isinstance(ctx, evalctx.IEEECtx):
        return mpmf.MPMF(m=1, exp=ctx.n + 1, ctx=ctx)
    elif isinstance(ctx, evalctx.PositCtx):
        return mpmf.MPMF(m=1, exp=ctx.emin, ctx=ctx)
    else:
        raise ValueError(f'unsupported type: {type(ctx)!r}')

def safe_mul_ctx(ctx):
    if isinstance(ctx, evalctx.IEEECtx):
        safe_es = ctx.es + 2
        safe_p = (ctx.p + 1) * 2
        return ieee754.ieee_ctx(safe_es, safe_es + safe_p)
    elif isinstance(ctx, evalctx.PositCtx):
        # very conservative; not a posit ctx
        log_emax = ctx.emax.bit_length()
        safe_es = log_emax + 2
        safe_p = (ctx.p + 1) * 2
        return ieee754.ieee_ctx(safe_es, safe_es + safe_p)
    else:
        raise ValueError(f'unsupported type: {type(ctx)!r}')

def safe_quire_ctx(ctx, log_carries = 30):
    mul_ctx = safe_mul_ctx(ctx)

    largest = largest_representable(ctx)
    largest_squared = largest.mul(largest, ctx=mul_ctx)

    smallest = smallest_representable(ctx)
    smallest_squared = smallest.mul(smallest, ctx=mul_ctx)

    # check
    assert largest_squared.inexact is False and smallest_squared.inexact is False

    left = largest_squared.e + 1 + log_carries
    right = smallest_squared.e

    quire_type = fixed.fixed_ctx(right, left - right)

    # check
    assert not fixed.Fixed._round_to_context(largest_squared, ctx=quire_type).isinf
    assert not fixed.Fixed._round_to_context(smallest_squared, ctx=quire_type).is_zero()

    return quire_type


def round_vec(v, ctx):
    return ndarray.NDArray([mpmf.MPMF(x, ctx=ctx) for x in v])

def rand_vec(n, ctx=None, signed=True):
    if signed:
        v = [random.random() if random.randint(0,1) else -random.random() for _ in range(n)]
    else:
        v = [random.random() for _ in range(n)]

    if ctx is None:
        return v
    else:
        return round_vec(v, ctx)


def setup_dotprod(template, precs):
    evaltor = Interpreter()
    main = load_cores(evaltor, mk_dotprod(template, *precs))
    return evaltor, main

def setup_full_quire(ctx, unrounded=False):
    qctx = safe_quire_ctx(ctx)
    precs = (ctx.propstr(), '', qctx.propstr())
    if unrounded:
        template = dotprod_fused_unrounded_template
    else:
        template = dotprod_fused_template
    return setup_dotprod(template, precs)

# sweep
# constants: base dtype
#            # trials (input data...)
# variables: quire high bits
#            quire lo bits
# metrics:   ulps


# BAD - globals

class VecSettings(object):
    def __init__(self):
        self.trials = None
        self.n = None
        self.As = None
        self.Bs = None
        self.refs = None
        self.template = None
        self.overall_ctx = None
        self.mul_ctx = None

    def cfg(self, trials, n, ctx, template, signed=True):
        self.trials = trials
        self.n = n
        self.As = [rand_vec(n, ctx=ctx, signed=signed) for _ in range(trials)]
        self.Bs = [rand_vec(n, ctx=ctx, signed=signed) for _ in range(trials)]
        evaltor, main = setup_full_quire(ctx)
        self.refs = [evaltor.interpret(main, [a, b]) for a, b in zip(self.As, self.Bs)]
        self.template = template
        self.overall_ctx = ctx
        self.mul_ctx = safe_mul_ctx(ctx)

        print(mk_dotprod(template, self.overall_ctx.propstr(), self.mul_ctx.propstr(), safe_quire_ctx(ctx).propstr()))

global_settings = VecSettings()


def vec_stage(quire_lo, quire_hi):
    try:
        overall_prec = global_settings.overall_ctx.propstr()
        mul_prec = global_settings.mul_ctx.propstr()
        sum_prec = fixed.fixed_ctx(-quire_lo, quire_lo + quire_hi).propstr()
        precs = (overall_prec, mul_prec, sum_prec)
        evaltor, main = setup_dotprod(global_settings.template, precs)

        worst_ulps = 0
        sum_ulps = 0
        infs = 0

        for a, b, ref in zip(global_settings.As, global_settings.Bs, global_settings.refs):
            result = evaltor.interpret(main, [a, b])
            if result.is_finite_real():
                ulps = abs(linear_ulps(result, ref))
                sum_ulps += ulps
                if ulps > worst_ulps:
                    worst_ulps = ulps
            else:
                worst_ulps = math.inf
                sum_ulps = math.inf
                infs += 1

        avg_ulps = sum_ulps / global_settings.trials

        return quire_lo + quire_hi, infs, worst_ulps, avg_ulps
    except Exception:
        traceback.print_exc()
        return math.inf, math.inf, math.inf, math.inf

def init_prec():
    return 16
def neighbor_prec(x):
    nearby = 5
    for neighbor in range(x-nearby, x+nearby+1):
        if 1 <= neighbor <= 4096 and neighbor != x:
            yield neighbor

vec_inits = (init_prec,) * 2
vec_neighbors = (neighbor_prec,) * 2
vec_metrics = (operator.lt,) * 3

def run_sweep(trials, n, ctx, template, signed=True):
    global_settings.cfg(trials, n, ctx, template, signed=signed)
    search.sweep_random_init(vec_stage, vec_inits, vec_neighbors, vec_metrics)





bf16 = ieee754.ieee_ctx(8, 16)
f16 = ieee754.ieee_ctx(5, 16)
p16 = posit.posit_ctx(0, 16)
p16_1 = posit.posit_ctx(1, 16)










# some results
from math import inf as inf

# run_sweep(100, 1000, bf16, dotprod_naive_template)
stuff = [
    ((16, 5), (21, 0, 12, 0.57)),
    ((17, 5), (22, 0, 4, 0.19)),
    ((18, 5), (23, 0, 1, 0.04)),
    ((20, 5), (25, 0, 0, 0.0)),
    ((11, 5), (16, 0, 1928, 60.58)),
    ((13, 5), (18, 0, 1660, 31.54)),
    ((14, 5), (19, 0, 190, 5.18)),
    ((15, 5), (20, 0, 72, 2.01)),
    ((4, 6), (10, 0, 125090, 5556.55)),
    ((5, 6), (11, 0, 64674, 3095.17)),
    ((7, 5), (12, 0, 16034, 803.76)),
    ((8, 5), (13, 0, 8034, 500.84)),
    ((10, 5), (15, 0, 1970, 93.25)),
    ((1, 1), (2, 100, inf, inf)),
    ((1, 8), (9, 23, inf, inf)),
    ((5, 3), (8, 94, inf, inf)),
]

# run_sweep(100, 1000, bf16, dotprod_bin_template)
# improvement stopped at generation 7: 
stuff = [
    ((16, 5), (21, 0, 5, 0.17)),
    ((17, 5), (22, 0, 2, 0.07)),
    ((18, 5), (23, 0, 1, 0.03)),
    ((21, 5), (26, 0, 0, 0.0)),
    ((11, 5), (16, 0, 247, 8.6)),
    ((12, 5), (17, 0, 125, 3.98)),
    ((13, 5), (18, 0, 61, 1.86)),
    ((14, 5), (19, 0, 30, 0.86)),
    ((15, 5), (20, 0, 13, 0.42)),
    ((3, 6), (9, 0, 62815, 2184.57)),
    ((4, 6), (10, 0, 32479, 1237.56)),
    ((6, 5), (11, 0, 15744, 733.06)),
    ((7, 5), (12, 0, 3887, 162.52)),
    ((8, 5), (13, 0, 1983, 81.96)),
    ((10, 5), (15, 0, 503, 19.36)),
    ((1, 1), (2, 100, inf, inf)),
    ((1, 7), (8, 40, inf, inf)),
    ((4, 3), (7, 88, inf, inf)),
]


# run_sweep(100, 1000, p16, dotprod_naive_template)
# improvement stopped at generation 7: 
stuff = [
    ((16, 5), (21, 0, 127, 17.63)),
    ((17, 5), (22, 0, 64, 8.68)),
    ((18, 5), (23, 0, 32, 4.31)),
    ((19, 5), (24, 0, 16, 2.15)),
    ((20, 5), (25, 0, 8, 1.02)),
    ((21, 5), (26, 0, 4, 0.47)),
    ((6, 6), (12, 0, 129717, 28580.59)),
    ((11, 5), (16, 0, 4084, 603.34)),
    ((12, 5), (17, 0, 2032, 298.69)),
    ((13, 5), (18, 0, 1029, 146.11)),
    ((14, 5), (19, 0, 510, 73.76)),
    ((15, 5), (20, 0, 258, 36.43)),
    ((22, 5), (27, 0, 2, 0.2)),
    ((23, 5), (28, 0, 1, 0.06)),
    ((26, 5), (31, 0, 0, 0.0)),
    ((4, 6), (10, 0, 517871, 73538.57)),
    ((5, 6), (11, 0, 259311, 47042.38)),
    ((8, 5), (13, 0, 33455, 6982.4)),
    ((9, 5), (14, 0, 16495, 2870.68)),
    ((10, 5), (15, 0, 8143, 1276.26)),
    ((1, 1), (2, 100, inf, inf)),
    ((1, 8), (9, 37, inf, inf)),
    ((4, 3), (7, 99, inf, inf)),
    ((4, 4), (8, 96, inf, inf)),
]

# run_sweep(100, 1000, p16, dotprod_bin_template)
# improvement stopped at generation 7: 
stuff = [
    ((16, 6), (22, 0, 64, 10.26)),
    ((17, 6), (23, 0, 32, 5.14)),
    ((18, 6), (24, 0, 16, 2.57)),
    ((19, 6), (25, 0, 9, 1.25)),
    ((20, 6), (26, 0, 4, 0.63)),
    ((21, 6), (27, 0, 2, 0.31)),
    ((11, 6), (17, 0, 2085, 332.43)),
    ((12, 6), (18, 0, 1087, 166.34)),
    ((13, 6), (19, 0, 547, 82.46)),
    ((14, 6), (20, 0, 261, 41.09)),
    ((15, 6), (21, 0, 130, 20.56)),
    ((22, 6), (28, 0, 1, 0.13)),
    ((26, 6), (32, 0, 0, 0.0)),
    ((3, 6), (9, 0, 539674, 91906.36)),
    ((6, 5), (11, 0, 67363, 16120.78)),
    ((7, 5), (12, 0, 34520, 6755.21)),
    ((8, 5), (13, 0, 16794, 3023.6)),
    ((9, 5), (14, 0, 8340, 1413.98)),
    ((10, 5), (15, 0, 4180, 681.31)),
    ((1, 1), (2, 100, inf, inf)),
    ((1, 7), (8, 42, inf, inf)),
    ((3, 3), (6, 99, inf, inf)),
    ((4, 3), (7, 91, inf, inf)),
    ((5, 5), (10, 0, 136730, 28892.4)),
]

