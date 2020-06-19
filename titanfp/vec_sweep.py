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
          (! {sum_prec} (ref A k1))))
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
 {overall_prec}
  (let ([result (binsum (vec-prod A B))])
    (cast result)))
''')

dotprod_neumaier_template = (
    nksum_template + '\n' +
    vec_prod_template + '\n' +
'''(FPCore dotprod ((A n) (B m))
 :pre (== n m)
 {overall_prec}
  (let ([result (nksum (vec-prod A B))])
    (cast result)))
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
        self.n = None
        self.As = None
        self.Bs = None
        self.refs = None
        self.template = None
        self.overall_ctx = None
        self.mul_ctx = None

    def cfg(self, n, ctx, template, signed=True):
        self.n = n
        self.As = [rand_vec(n, ctx=ctx, signed=signed) for _ in range(n)]
        self.Bs = [rand_vec(n, ctx=ctx, signed=signed) for _ in range(n)]
        evaltor, main = setup_full_quire(ctx)
        self.refs = [evaltor.interpret(main, [a, b]) for a, b in zip(self.As, self.Bs)]
        self.template = template
        self.overall_ctx = ctx
        self.mul_ctx = safe_mul_ctx(ctx)

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

        avg_ulps = sum_ulps / global_settings.n

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

def run_sweep(n, ctx, template, signed=True):
    global_settings.cfg(n, ctx, template, signed=signed)
    search.sweep_random_init(vec_stage, vec_inits, vec_neighbors, vec_metrics)


bf16 = ieee754.ieee_ctx(8, 16)
f16 = ieee754.ieee_ctx(5, 16)
p16 = posit.posit_ctx(0, 16)
p16_1 = posit.posit_ctx(1, 16)


