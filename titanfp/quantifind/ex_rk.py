"""Runge-Kutta stepper for chaotic attractors."""

import operator
import math
import traceback

from ..titanic import ndarray, gmpmath
from ..fpbench import fpcparser
from ..arithmetic import mpmf, ieee754, posit, analysis

from . import search
from .utils import *
from .benchmarks import mk_rk, rk_equations, rk_data



def worst_and_avg_abits(a1, a2, ctx):
    worst_abits = math.inf
    total_abits = 0

    for e1, e2 in zip(a1, a2):
        if e1.is_finite_real() and e2.is_finite_real():
            if ctx is None:
                abits = gmpmath.geo_sim(e1, e2)
            else:
                abits = min(gmpmath.geo_sim(e1, e2), ctx.p)
            if abits < worst_abits:
                worst_abits = abits
            total_abits += abits
        else:
            worst_abits = -math.inf
            total_abits = -math.inf
    return worst_abits, total_abits / min(len(a1), len(a2))


def setup_rk(ebits, fn_prec, rk_prec, k1_prec, k2_prec, k3_prec, k4_prec,
             method='rk4', eqn=None, use_posit=False):
    if eqn is None:
        raise ValueError('must specify an equation')

    if use_posit:
        mk_ctx = posit.posit_ctx
        extra_bits = 0
    else:
        mk_ctx = ieee754.ieee_ctx
        extra_bits = ebits

    fn_ctx = mk_ctx(ebits, fn_prec + extra_bits)
    rk_ctx = mk_ctx(ebits, rk_prec + extra_bits)
    k1_ctx = mk_ctx(ebits, k1_prec + extra_bits)
    k2_ctx = mk_ctx(ebits, k2_prec + extra_bits)
    k3_ctx = mk_ctx(ebits, k3_prec + extra_bits)
    k4_ctx = mk_ctx(ebits, k4_prec + extra_bits)

    prog = mk_rk(fn_ctx, rk_ctx, k1_ctx, k2_ctx, k3_ctx, k4_ctx,
                 method=method, eqn=eqn)

    eqn_name, eqn_template = rk_equations[eqn]
    equation = eqn_template.format(fn_prec=fn_ctx.propstr())

    return fpcparser.compile(prog), fpcparser.compile(equation), rk_ctx

def run_rk(prog, args):
    evaltor = mpmf.Interpreter()
    als = analysis.BitcostAnalysis()
    main = load_cores(evaltor, prog, [als])

    result_array = evaltor.interpret(main, args)
    return evaltor, als, result_array

def eval_rk(equation, als, result_array, ref, dref, ctx=None):
    last = result_array[-1]

    evaltor = mpmf.Interpreter()
    main = load_cores(evaltor, equation)
    dlast = evaltor.interpret(main, [ndarray.NDArray(last)])

    worst_abits_last, avg_abits_last = worst_and_avg_abits(last, ref, ctx)
    worst_abits_dlast, avg_abits_dlast = worst_and_avg_abits(dlast, dref, ctx)

    return als.bits_requested, worst_abits_last, avg_abits_last, worst_abits_dlast, avg_abits_dlast


class RkSettings(object):
    """Global settings"""

    def __init__(self):
        self.method = 'rk4'
        self.eqn = 'lorenz'
        args, ref, dref = rk_data[self.eqn]
        self.args = fpcparser.read_exprs(args)
        self.ref = [mpmf.MPMF(e, ctx=f64) for e in ref]
        self.dref = [mpmf.MPMF(e, ctx=f64) for e in dref]
        self.use_posit = False

    def cfg(self, eqn, use_posit):
        self.eqn = eqn
        args, ref, dref = rk_data[self.eqn]
        self.args = fpcparser.read_exprs(args)
        self.ref = [mpmf.MPMF(e, ctx=f64) for e in ref]
        self.dref = [mpmf.MPMF(e, ctx=f64) for e in dref]
        self.use_posit = use_posit

settings = RkSettings()


def rk_stage(ebits, fn_prec, rk_prec, k1_prec, k2_prec, k3_prec, k4_prec):
    try:
        prog, equation, ctx = setup_rk(ebits, fn_prec, rk_prec, k1_prec, k2_prec, k3_prec, k4_prec,
                                                        method=settings.method, eqn=settings.eqn, use_posit=settings.use_posit)
        evaltor, als, result_array = run_rk(prog, settings.args)
        return eval_rk(equation, als, result_array, settings.ref, settings.dref, ctx)
    except Exception:
        traceback.print_exc()
        return math.inf, -math.inf, -math.inf, -math.inf, -math.inf

def rk_ref_stage(fn_ctx, rk_ctx, k1_ctx, k2_ctx, k3_ctx, k4_ctx):
    try:
        prog = mk_rk(fn_ctx, rk_ctx, k1_ctx, k2_ctx, k3_ctx, k4_ctx,
                     method=settings.method, eqn=settings.eqn)
        eqn_name, eqn_template = rk_equations[settings.eqn]
        equation = eqn_template.format(fn_prec=fn_ctx.propstr())
        prog, equation, ctx = fpcparser.compile(prog), fpcparser.compile(equation), rk_ctx

        evaltor, als, result_array = run_rk(prog, settings.args)
        return eval_rk(equation, als, result_array, settings.ref, settings.dref, ctx)
    except Exception:
        traceback.print_exc()
        return math.inf, -math.inf, -math.inf, -math.inf, -math.inf

def rk_fenceposts():
    points = [
        ((describe_ctx(ctx),), rk_ref_stage(*((ctx,) * 6)))
        for ctx in float_basecase + posit_basecase
    ]

    return [0], [(0, a, b) for a, b in points], points

def rk_ceiling():
    ceil_pts = []
    settings.cfg('lorenz', False)
    ceil_pts.append(rk_ref_stage(*((f4k,) * 6)))
    settings.cfg('rossler', False)
    ceil_pts.append(rk_ref_stage(*((f4k,) * 6)))
    settings.cfg('chua', False)
    ceil_pts.append(rk_ref_stage(*((f4k,) * 6)))

    for record in ceil_pts:
        print(repr(record))

    return ceil_pts


def rk_experiment(prefix, ebit_slice, pbit_slice, es_slice, inits, retries):
    rk_metrics = (operator.lt,) + (operator.gt,) * 4

    init_ebits, neighbor_ebits = integer_neighborhood(*ebit_slice)
    init_pbits, neighbor_pbits = integer_neighborhood(*pbit_slice)

    # for posits
    init_es, neighbor_es = integer_neighborhood(*es_slice)

    rk_inits = (init_ebits,) + (init_pbits,) * 6
    rk_neighbors = (neighbor_ebits,) + (neighbor_pbits,) * 6

    settings.cfg('lorenz', False)
    try:
        sweep = search.sweep_multi(rk_stage, rk_inits, rk_neighbors, rk_metrics, inits, retries, force_exploration=True)
        jsonlog(prefix + '_rk_lorenz.json', *sweep, settings='Lorenz with floats')
    except Exception:
        traceback.print_exc()

    settings.cfg('rossler', False)
    try:
        sweep = search.sweep_multi(rk_stage, rk_inits, rk_neighbors, rk_metrics, inits, retries, force_exploration=True)
        jsonlog(prefix + '_rk_rossler.json', *sweep, settings='Rossler with floats')
    except Exception:
        traceback.print_exc()

    settings.cfg('chua', False)
    try:
        sweep = search.sweep_multi(rk_stage, rk_inits, rk_neighbors, rk_metrics, inits, retries, force_exploration=True)
        jsonlog(prefix + '_rk_chua.json', *sweep, settings='Chua with floats')
    except Exception:
        traceback.print_exc()

    rk_inits = (init_es,) + (init_pbits,) * 6
    rk_neighbors = (neighbor_es,) + (neighbor_pbits,) * 6

    settings.cfg('lorenz', True)
    try:
        sweep = search.sweep_multi(rk_stage, rk_inits, rk_neighbors, rk_metrics, inits, retries, force_exploration=True)
        jsonlog(prefix + '_rk_lorenz_p.json', *sweep, settings='Lorenz with posits')
    except Exception:
        traceback.print_exc()

    settings.cfg('rossler', True)
    try:
        sweep = search.sweep_multi(rk_stage, rk_inits, rk_neighbors, rk_metrics, inits, retries, force_exploration=True)
        jsonlog(prefix + '_rk_rossler_p.json', *sweep, settings='Rossler with posits')
    except Exception:
        traceback.print_exc()

    settings.cfg('chua', True)
    try:
        sweep = search.sweep_multi(rk_stage, rk_inits, rk_neighbors, rk_metrics, inits, retries, force_exploration=True)
        jsonlog(prefix + '_rk_chua_p.json', *sweep, settings='Chua with posits')
    except Exception:
        traceback.print_exc()


def rk_baseline(prefix):
    rk_bc_float = (float_basecase,) * 6
    rk_bc_posit = (posit_basecase,) * 6
    rk_metrics = (operator.lt,) + (operator.gt,) * 4

    settings.cfg('lorenz', False)
    try:
        sweep = search.sweep_exhaustive(rk_ref_stage, rk_bc_float, rk_metrics)
        jsonlog(prefix + '_rk_lorenz.json', *sweep, settings='Lorenz with floats baseline')
    except Exception:
        traceback.print_exc()

    try:
        sweep = rk_fenceposts()
        jsonlog(prefix + '_rk_lorenz_fenceposts.json', *sweep, settings='Lorenz with floats fenceposts')
    except Exception:
        traceback.print_exc()

    settings.cfg('rossler', False)
    try:
        sweep = search.sweep_exhaustive(rk_ref_stage, rk_bc_float, rk_metrics)
        jsonlog(prefix + '_rk_rossler.json', *sweep, settings='Rossler with floats baseline')
    except Exception:
        traceback.print_exc()

    try:
        sweep = rk_fenceposts()
        jsonlog(prefix + '_rk_rossler_fenceposts.json', *sweep, settings='Rossler with floats fenceposts')
    except Exception:
        traceback.print_exc()

        
    settings.cfg('chua', False)
    try:
        sweep = search.sweep_exhaustive(rk_ref_stage, rk_bc_float, rk_metrics)
        jsonlog(prefix + '_rk_chua.json', *sweep, settings='Chua with floats baseline')
    except Exception:
        traceback.print_exc()

    try:
        sweep = rk_fenceposts()
        jsonlog(prefix + '_rk_chua_fenceposts.json', *sweep, settings='Chua with floats fenceposts')
    except Exception:
        traceback.print_exc()

    # again with posits
        
    settings.cfg('lorenz', True)
    try:
        sweep = search.sweep_exhaustive(rk_ref_stage, rk_bc_posit, rk_metrics)
        jsonlog(prefix + '_rk_lorenz_p.json', *sweep, settings='Lorenz with posits baseline')
    except Exception:
        traceback.print_exc()

    try:
        sweep = rk_fenceposts()
        jsonlog(prefix + '_rk_lorenz_p_fenceposts.json', *sweep, settings='Lorenz with posits fenceposts')
    except Exception:
        traceback.print_exc()

    settings.cfg('rossler', True)
    try:
        sweep = search.sweep_exhaustive(rk_ref_stage, rk_bc_posit, rk_metrics)
        jsonlog(prefix + '_rk_rossler_p.json', *sweep, settings='Rossler with posits baseline')
    except Exception:
        traceback.print_exc()

    try:
        sweep = rk_fenceposts()
        jsonlog(prefix + '_rk_rossler_p_fenceposts.json', *sweep, settings='Rossler with posits fenceposts')
    except Exception:
        traceback.print_exc()

        
    settings.cfg('chua', True)
    try:
        sweep = search.sweep_exhaustive(rk_ref_stage, rk_bc_posit, rk_metrics)
        jsonlog(prefix + '_rk_chua_p.json', *sweep, settings='Chua with posits baseline')
    except Exception:
        traceback.print_exc()

    try:
        sweep = rk_fenceposts()
        jsonlog(prefix + '_rk_chua_p_fenceposts.json', *sweep, settings='Chua with posits fenceposts')
    except Exception:
        traceback.print_exc()
