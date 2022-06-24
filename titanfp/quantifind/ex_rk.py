"""Runge-Kutta stepper for chaotic attractors."""

import operator
import math
import traceback

from ..titanic import ndarray, gmpmath
from ..fpbench import fpcparser
from ..arithmetic import mpmf, ieee754, posit, analysis

from . import search
from .utils import *
from .benchmarks import mk_rk, rk_equations, rk_data, rk_step_sizes



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
        step_sizes = rk_step_sizes[self.eqn]
        self.args = fpcparser.read_exprs(args)
        self.ref = [mpmf.MPMF(e, ctx=f64) for e in ref]
        self.dref = [mpmf.MPMF(e, ctx=f64) for e in dref]
        self.use_posit = False

    def cfg(self, eqn, use_posit):
        self.eqn = eqn
        args, ref, dref = rk_data[self.eqn]
        step_sizes = rk_step_sizes[self.eqn]
        self.args = fpcparser.read_exprs(args)
        self.ref = [mpmf.MPMF(e, ctx=f64) for e in ref]
        self.dref = [mpmf.MPMF(e, ctx=f64) for e in dref]
        self.step_sizes = [fpcparser.read_exprs(e) for e in step_sizes]
        self.use_posit = use_posit

settings = RkSettings()


def rk_stage(ebits, fn_prec, rk_prec, k1_prec, k2_prec, k3_prec, k4_prec, step_size=None):
    try:
        prog, equation, ctx = setup_rk(ebits, fn_prec, rk_prec, k1_prec, k2_prec, k3_prec, k4_prec,
                                                        method=settings.method, eqn=settings.eqn, use_posit=settings.use_posit)
        args = settings.args
        if step_size is not None:
            args = (args[0], *settings.step_sizes[step_size])
        evaltor, als, result_array = run_rk(prog, args)

        static_bits = fn_prec + rk_prec + k1_prec + k2_prec + k3_prec + k4_prec
        if not settings.use_posit:
            static_bits += ebits * 6

        return (static_bits, *eval_rk(equation, als, result_array, settings.ref, settings.dref, ctx))
    except Exception:
        traceback.print_exc()
        return (math.inf, math.inf, -math.inf, -math.inf, -math.inf, -math.inf)

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


def rk_experiment(prefix, ebit_slice, pbit_slice, es_slice, inits, retries, use_step_sizes=False, eq_name='all'):
    if eq_name == 'all':
        rk_experiment(prefix, ebit_slice, pbit_slice, es_slice, inits, retries, eq_name='lorenz')
        rk_experiment(prefix, ebit_slice, pbit_slice, es_slice, inits, retries, eq_name='rossler')
        rk_experiment(prefix, ebit_slice, pbit_slice, es_slice, inits, retries, eq_name='chua')
        return
    elif eq_name not in ['lorenz', 'rossler', 'chua']:
        print(f'Unknown equation {eq_name}')
        return

    rk_metrics = (operator.lt,) * 2 + (operator.gt,) * 4

    init_ebits, neighbor_ebits = integer_neighborhood(*ebit_slice)
    init_pbits, neighbor_pbits = integer_neighborhood(*pbit_slice)

    # for posits
    init_es, neighbor_es = integer_neighborhood(*es_slice)

    if use_step_sizes:
        init_steps, neighbor_steps = integer_neighborhood(0, 13, 2)

    rk_inits = (init_ebits,) + (init_pbits,) * 6
    rk_neighbors = (neighbor_ebits,) + (neighbor_pbits,) * 6
    if use_step_sizes:
        rk_inits += (init_steps,)
        rk_neighbors += (neighbor_steps,)

    cores = 64
    sweep_settings = search.SearchSettings(
        profile = 'balanced',
        initial_gen_size = cores * inits,
        restart_gen_target = retries,
        pop_targets = [
            (cores,   None),
            (cores,   None),
            (cores,   None),
            (cores*3, None),
        ]
    )

    cores = 128

    settings.cfg(eq_name, False)
    try:
        with search.Sweep(rk_stage, rk_inits, rk_neighbors, rk_metrics, settings=sweep_settings, cores=cores) as sweep:
            frontier = sweep.run_search(checkpoint_dir=prefix+'/float')
            sweepdata = sweep.state.generations, sweep.state.history, frontier
        #sweep = search.sweep_multi(rk_stage, rk_inits, rk_neighbors, rk_metrics, inits, retries, force_exploration=True)
        #jsonlog(prefix + '_rk_' + eq_name + '.json', *sweepdata, settings=eq_name + ' with floats')
    except Exception:
        traceback.print_exc()

    if True: # skip posit experiment
        return

    rk_inits = (init_es,) + (init_pbits,) * 6
    rk_neighbors = (neighbor_es,) + (neighbor_pbits,) * 6
    if use_step_sizes:
        rk_inits += (init_steps,)
        rk_neighbors += (neighbor_steps,)

    settings.cfg(eq_name, True)
    try:
        with search.Sweep(rk_stage, rk_inits, rk_neighbors, rk_metrics, settings=sweep_settings, cores=cores) as sweep:
            frontier = sweep.run_search(checkpoint_dir=prefix+'/posit')
            sweepdata = sweep.state.generations, sweep.state.history, frontier
        #sweep = search.sweep_multi(rk_stage, rk_inits, rk_neighbors, rk_metrics, inits, retries, force_exploration=True)
        #jsonlog(prefix + '_rk_' + eq_name + '_p.json', *sweepdata, settings=eq_name + ' with posits')
    except Exception:
        traceback.print_exc()

def rk_random(prefix, ebit_slice, pbit_slice, es_slice, points, eq_name='all'):
    if eq_name == 'all':
        rk_experiment(prefix, ebit_slice, pbit_slice, es_slice, inits, retries, eq_name='lorenz')
        rk_experiment(prefix, ebit_slice, pbit_slice, es_slice, inits, retries, eq_name='rossler')
        rk_experiment(prefix, ebit_slice, pbit_slice, es_slice, inits, retries, eq_name='chua')
        return
    elif eq_name not in ['lorenz', 'rossler', 'chua']:
        print(f'Unknown equation {eq_name}')
        return

    rk_metrics = (operator.lt,) + (operator.gt,) * 4

    init_ebits, neighbor_ebits = integer_neighborhood(*ebit_slice)
    init_pbits, neighbor_pbits = integer_neighborhood(*pbit_slice)

    # for posits
    init_es, neighbor_es = integer_neighborhood(*es_slice)

    rk_inits = (init_ebits,) + (init_pbits,) * 6
    rk_neighbors = (neighbor_ebits,) + (neighbor_pbits,) * 6

    settings.cfg(eq_name, False)
    try:
        sweep = search.sweep_random(rk_stage, rk_inits, rk_metrics, points)
        jsonlog(prefix + '_random_rk_' + eq_name + '.json', *sweep, settings=eq_name + ' random with floats')
    except Exception:
        traceback.print_exc()

    rk_inits = (init_es,) + (init_pbits,) * 6
    rk_neighbors = (neighbor_es,) + (neighbor_pbits,) * 6

    settings.cfg(eq_name, True)
    try:
        sweep = search.sweep_random(rk_stage, rk_inits, rk_metrics, points)
        jsonlog(prefix + '_random_rk_' + eq_name + '_p.json', *sweep, settings=eq_name + ' random with posits')
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
