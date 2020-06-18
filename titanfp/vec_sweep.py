import math
import random
import operator

from .titanic import ndarray
from .fpbench import fpcparser
from .arithmetic import mpmf, ieee754, posit, fixed, evalctx, analysis
from .arithmetic.mpmf import Interpreter

from .sweep import search

dotprod_core = '''(FPCore dotprod ((A n) (B m))
 :pre (== n m)
  (for ([i n])
    ([prod 0 (+ prod (* (ref A i) (ref B i)))])
    prod))
'''

# need quire sizing

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

def rand_vec(n, ctx, signed=True):
    if signed:
        v = [random.random() if random.randint(0,1) else -random.random() for _ in range(n)]
    else:
        v = [random.random() for _ in range(n)]

    return ndarray.NDArray([mpmf.MPMF(x, ctx=ctx) for x in v])
