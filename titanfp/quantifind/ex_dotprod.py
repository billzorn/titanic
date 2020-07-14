"""Dot products of random vectors."""

import operator
import math
import random
import traceback

from ..titanic import ndarray, gmpmath
from ..fpbench import fpcparser
from ..arithmetic import mpmf, ieee754, posit, fixed, evalctx, analysis

from . import search
from .utils import *
from .benchmarks import mk_dotprod


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

def describe_vec(v):
    return '(' + ', '.join(str(e) for e in v) + ')'


def setup_dotprod(template, ctxs):
    evaltor = mpmf.Interpreter()
    main = load_cores(evaltor, mk_dotprod(template, *ctxs))
    return evaltor, main

def setup_full_quire(ctx, unrounded=False):
    if unrounded:
        template = 'unrounded'
    else:
        template = 'fused'
    ctxs = ctx, None, safe_quire_ctx(ctx)
    return setup_dotprod(template, ctxs)


class DotprodSettings(object):
    def __init__(self):
        self.trials = None
        self.n = None
        self.signed = None
        self.As = None
        self.Bs = None
        self.refs = None
        self.real_refs = None
        self.template = None
        self.overall_ctx = None
        self.mul_ctx = None

    def cfg(self, trials, n, ctx, template, signed=True):
        self.trials = trials
        self.n = n
        self.signed = signed
        self.As = [rand_vec(n, ctx=ctx, signed=signed) for _ in range(trials)]
        self.Bs = [rand_vec(n, ctx=ctx, signed=signed) for _ in range(trials)]
        evaltor, main = setup_full_quire(ctx, unrounded=False)
        self.refs = [evaltor.interpret(main, [a, b]) for a, b in zip(self.As, self.Bs)]
        evaltor, main = setup_full_quire(ctx, unrounded=True)
        self.real_refs = [evaltor.interpret(main, [a, b]) for a, b in zip(self.As, self.Bs)]
        self.template = template
        self.overall_ctx = ctx
        self.mul_ctx = safe_mul_ctx(ctx)

        print(mk_dotprod(template, self.overall_ctx, self.mul_ctx, safe_quire_ctx(ctx)))

    def describe_cfg(self):
        return (f'cfg({self.trials!r}, {self.n!r}, {self.overall_ctx!r}, {self.template!r}, signed={self.signed!r})\n'
                f'#As = [{", ".join(describe_vec(a) for a in self.As)}]\n'
                f'#Bs = [{", ".join(describe_vec(b) for b in self.Bs)}]')

settings = DotprodSettings()


def describe_stage(quire_lo, quire_hi):
    sum_ctx = fixed.fixed_ctx(-quire_lo, quire_lo + quire_hi)
    ctxs = settings.overall_ctx, settings.mul_ctx, sum_ctx
    print(mk_dotprod(settings.template, *ctxs))

def dotprod_stage(quire_lo, quire_hi):
    try:
        sum_ctx = fixed.fixed_ctx(-quire_lo, quire_lo + quire_hi)
        ctxs = settings.overall_ctx, settings.mul_ctx, sum_ctx
        evaltor, main = setup_dotprod(settings.template, ctxs)

        worst_ulps = 0
        total_ulps = 0
        worst_abits = math.inf
        total_abits = 0
        infs = 0

        for a, b, ref, real_ref in zip(settings.As, settings.Bs, settings.refs, settings.real_refs):
            result = evaltor.interpret(main, [a, b])
            if result.is_finite_real():
                ulps = abs(linear_ulps(result, ref))
                total_ulps += ulps
                if ulps > worst_ulps:
                    worst_ulps = ulps
                abits = min(gmpmath.geo_sim(result, real_ref), settings.overall_ctx.p)
                total_abits += abits
                if abits < worst_abits:
                    worst_abits = abits
            else:
                worst_ulps = math.inf
                total_ulps = math.inf
                worst_abits = -math.inf
                total_abits = -math.inf
                infs += 1

        avg_ulps = total_ulps / settings.trials
        avg_abits = total_abits / settings.trials

        return quire_lo + quire_hi, infs, worst_ulps, avg_ulps, worst_abits, avg_abits
    except Exception:
        traceback.print_exc()
        return math.inf, math.inf, math.inf, math.inf, -math.inf, -math.inf


def dotprod_experiment(prefix, quire_slice, quire_init_range, trials, n, inits, retries):
    init_bits, neighbor_bits = integer_neighborhood(*quire_slice)
    # we can allow the search to explore large quire sizes,
    # but we probably don't want to start there
    init_bits = lambda : random.randint(*quire_init_range)

    dotprod_inits = (init_bits,) * 2
    dotprod_neighbors = (neighbor_bits,) * 2
    dotprod_metrics = (operator.lt,) * 4 + (operator.gt,) * 2

    settings.cfg(trials, n, bf16, 'fused', signed=True)
    try:
        sweep = search.sweep_multi(dotprod_stage, dotprod_inits, dotprod_neighbors, dotprod_metrics, inits, retries, force_exploration=True)
        jsonlog(prefix + '_dotprod_fused.json', *sweep, settings=settings.describe_cfg())
    except Exception:
        traceback.print_exc()

    settings.cfg(trials, n, bf16, 'fused', signed=False)
    try:
        sweep = search.sweep_multi(dotprod_stage, dotprod_inits, dotprod_neighbors, dotprod_metrics, inits, retries, force_exploration=True)
        jsonlog(prefix + '_dotprod_fused_unsigned.json', *sweep, settings=settings.describe_cfg())
    except Exception:
        traceback.print_exc()

    settings.cfg(trials, n, bf16, 'bin', signed=True)
    try:
        sweep = search.sweep_multi(dotprod_stage, dotprod_inits, dotprod_neighbors, dotprod_metrics, inits, retries, force_exploration=True)
        jsonlog(prefix + '_dotprod_bin.json', *sweep, settings=settings.describe_cfg())
    except Exception:
        traceback.print_exc()

    settings.cfg(trials, n, bf16, 'bin', signed=False)
    try:
        sweep = search.sweep_multi(dotprod_stage, dotprod_inits, dotprod_neighbors, dotprod_metrics, inits, retries, force_exploration=True)
        jsonlog(prefix + '_dotprod_bin_unsigned.json', *sweep, settings=settings.describe_cfg())
    except Exception:
        traceback.print_exc()
