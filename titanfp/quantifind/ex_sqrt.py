"""Square root example with Newton's method."""

import operator
import math

from ..titanic import gmpmath
from ..fpbench import fpcparser
from ..arithmetic import mpmf, ieee754, evalctx, analysis

from . import search
from .utils import *
from .benchmarks import mk_sqrt


def setup_reference():
    one = mpmf.MPMF(1, ctx=bf16)
    ten = mpmf.MPMF(10, ctx=bf16)
    current = one
    one_to_ten = []
    reference_results = []

    while current <= ten:
        one_to_ten.append(current)
        ref_hi = mpmf.MPMF(current, ctx=f64)
        reference_results.append((current.sqrt(ctx=bf16), ref_hi.sqrt(ctx=f64)))
        current = current.next_float()
    return one_to_ten, reference_results

class SqrtSettings(object):
    """Global settings"""

    def __init__(self):
        self.example_inputs, self.reference_outputs = setup_reference()
        self.overall_ctx = bf16
        (self.bound,) = fpcparser.read_exprs('1/100')
        self.babylonian = False

    def cfg(self, babylonian):
        self.babylonian = babylonian

settings = SqrtSettings()


def eval_sqrt(core, bound):
    timeouts = 0
    infs = 0
    worst_ulps = 0
    total_ulps = 0
    worst_abits = math.inf
    total_abits = 0
    worst_bitcost = 0
    total_bitcost = 0

    for arg, (ref, ref_hi) in zip(settings.example_inputs, settings.reference_outputs):
        evaltor = mpmf.Interpreter()
        als = analysis.BitcostAnalysis()
        evaltor.analyses = [als]
        result = evaltor.interpret(core, (arg, bound))

        steps = evaltor.evals
        bitcost = als.bits_requested

        if steps > 200:
            timeouts += 1

        if result.is_finite_real():
            ulps = abs(linear_ulps(result, ref))
            if ulps > worst_ulps:
                worst_ulps = ulps
            total_ulps += ulps
            abits = min(gmpmath.geo_sim(result, ref_hi), settings.overall_ctx.p)
            if abits < worst_abits:
                worst_abits = abits
            total_abits += abits
        else:
            worst_ulps = math.inf
            total_ulps = math.inf
            worst_abits = -math.inf
            total_abits = -math.inf
            infs += 1

        if bitcost > worst_bitcost:
            worst_bitcost = bitcost
        total_bitcost += bitcost

    return timeouts, infs, worst_bitcost, total_bitcost, worst_ulps, total_ulps, worst_abits, total_abits

def sqrt_stage(expbits, res_bits, diff_bits, scale_bits):
    prog = mk_sqrt(expbits, res_bits, diff_bits, scale_bits,
                   overall_ctx=settings.overall_ctx, babylonian=settings.babylonian)
    core = fpcparser.compile1(prog)
    return eval_sqrt(core, settings.bound)


def full_sweep(minexp, maxexp, minp, maxp, verbosity=3):
    from multiprocessing import Pool

    cfgs = 0
    all_cfgs = set()
    frontier = []
    metrics = (operator.lt,) * 6 + (operator.gt,) * 2

    with Pool() as p:
        result_buf = []

        for expbits in range(minexp,maxexp+1):
            for res_bits in range(minp, maxp+1):
                for diff_bits in range(minp, maxp+1):
                    for scale_bits in range(minp, maxp+1):
                        args = expbits, res_bits, diff_bits, scale_bits
                        result_buf.append((args, p.apply_async(sqrt_stage, args)))

        if verbosity >= 1:
            print(f'waiting for {len(result_buf)!s} results to come back')

        for args, async_result in result_buf:
            config = args
            result = async_result.get()

            cfgs += 1
            all_cfgs.add(config)
            if verbosity >= 3:
                print(f' -- {cfgs!s} -- ran {config!r}, got {result!r}')

            frontier_elt = (config, result)
            updated, frontier = search.update_frontier(frontier, frontier_elt, metrics)
            if updated and verbosity >= 2:
                print('The frontier changed:')
                search.print_frontier(frontier)

    if verbosity >= 1:
        print(f'tried {cfgs!s} configs, done\n')
        if verbosity >= 2:
            print('Final frontier:')
            search.print_frontier(frontier)
            print(flush=True)

    return [1], all_cfgs, frontier



def sqrt_test():
    init_expbits, neighboring_expbits = integer_neighborhood(3, 5, 2)
    init_sigbits, neighboring_sigbits = integer_neighborhood(5, 12, 3)

    sqrt_inits = (init_expbits,) + (init_sigbits,)*3
    sqrt_neighbors = (neighboring_expbits,) + (neighboring_sigbits,)*3
    sqrt_metrics = (operator.lt,) * 6 + (operator.gt,) * 2

    settings.cfg(False)

    sweep = full_sweep(3,5,7,8)
    jsonlog('test_newton_full.json', *sweep, settings='full sweep, newton')

    sweep = search.sweep_multi(sqrt_stage, sqrt_inits, sqrt_neighbors, sqrt_metrics, 20, 100, force_exploration=True)
    jsonlog('test_newton_random.json', *sweep, settings='random 20 100, newton')

    settings.cfg(True)

    sweep = full_sweep(3,5,7,9)
    jsonlog('test_babylonian_full.json', *sweep, settings='full sweep, babylonian')

    sweep = search.sweep_multi(sqrt_stage, sqrt_inits, sqrt_neighbors, sqrt_metrics, 20, 100, force_exploration=True)
    jsonlog('test_babylonian_random.json', *sweep, settings='random 20 100, babylonian')


def sqrt_experiment():
    init_expbits, neighboring_expbits = integer_neighborhood(2, 8, 2)
    init_sigbits, neighboring_sigbits = integer_neighborhood(1, 32, 3)

    sqrt_inits = (init_expbits,) + (init_sigbits,)*3
    sqrt_neighbors = (neighboring_expbits,) + (neighboring_sigbits,)*3
    sqrt_metrics = (operator.lt,) * 6 + (operator.gt,) * 2

    settings.cfg(False)

    sweep = full_sweep(3,5,2,20)
    jsonlog('sweep_newton_full.json', *sweep, settings='full sweep, newton')

    sweep = search.sweep_multi(sqrt_stage, sqrt_inits, sqrt_neighbors, sqrt_metrics, 20, 100, force_exploration=True)
    jsonlog('sweep_newton_random.json', *sweep, settings='random 20 100, newton')

    settings.cfg(True)

    sweep = full_sweep(3,5,2,20)
    jsonlog('sweep_babylonian_full.json', *sweep, settings='full sweep, babylonian')

    sweep = search.sweep_multi(sqrt_stage, sqrt_inits, sqrt_neighbors, sqrt_metrics, 20, 100, force_exploration=True)
    jsonlog('sweep_babylonian_random.json', *sweep, settings='random 20 100, babylonian')
