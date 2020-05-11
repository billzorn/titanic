import math
import operator

from .fpbench import fpcparser
from .arithmetic import mpmf, ieee754, evalctx, analysis
from .arithmetic.mpmf import Interpreter

from .sweep import search


sqrt_core = '''(FPCore sqrt_bfloat_limit (a residual_bound)
 :precision {overall_prec}

 (while* (and (< steps (# 20)) (>= (fabs residual) residual_bound))
  ([x a (! :precision {diff_prec} (- x
                                     (! :precision {scale_prec} (/ residual (* 2 x)))))]
   [residual (! :precision {res_prec} (- (* x x) a))
             (! :precision {res_prec} (- (* x x) a))]
   [steps 0 (# (+ 1 steps))])
  (cast x))

)
'''

formatted = sqrt_core.format(
    overall_prec = '(float 8 16)',
    res_prec = '(float 5 16)',
    diff_prec = '(float 5 16)',
    scale_prec = '(float 5 16)',
)

evaltor = Interpreter()
core = fpcparser.compile1(formatted)

bf16 = ieee754.ieee_ctx(8,16)
one = mpmf.MPMF(1, ctx=bf16)
ten = mpmf.MPMF(10, ctx=bf16)
current = one
one_to_ten = []
reference_results = []
while current <= ten:
    one_to_ten.append(current)
    reference_results.append(current.sqrt())
    current = current.next_float()

def run_tests(core, bound):

    results = []

    for arg, ref in zip(one_to_ten, reference_results):
        evaltor = Interpreter()
        als, bc_als = analysis.DefaultAnalysis(), analysis.BitcostAnalysis()
        #evaltor.analyses = [als, bc_als]
        result = evaltor.interpret(core, (arg, bound))
        err = (ref.sub(result)).fabs()
        steps = evaltor.evals

        results.append((str(arg), str(result), str(err), steps))

    return results

def oob(results):
    counted = 0
    maxerr = 0
    sumerr = 0
    sumcount = 0
    for arg, result, err, steps in results:
        if steps > 200:
            counted += 1
        abserr = abs(float(err))
        if math.isfinite(abserr):
            sumerr += abserr
            sumcount += 1
        # if abserr > 0.01:
        #     print(arg, result, err, steps)
        if abserr > maxerr:
            maxerr = abserr
    # print(f'{counted!s} inputs ran for more than 200 steps.')
    # print(f'worst point was {maxerr!s}.')
    return counted, maxerr, sumerr / sumcount


# new, with bitcost
def run_and_eval_bitcost(core, bound):

    results = []

    for arg, ref in zip(one_to_ten, reference_results):
        evaltor = Interpreter()
        als = analysis.BitcostAnalysis()
        evaltor.analyses = [als]
        result = evaltor.interpret(core, (arg, bound))
        err = (ref.sub(result)).fabs()
        steps = evaltor.evals
        bitcost = als.bits_requested

        abserr = abs(float(err))
        # if not math.isfinite(err):
        #     print(str(arg), str(result), str(err))

        results.append((str(arg), str(result), str(err), steps, bitcost))

    timeouts = 0
    infs = 0
    maxerr = 0
    worst_bitcost = 0
    total_bitcost = 0
    for arg, result, err, steps, bitcost in results:
        if steps > 200:
            timeouts += 1

        if bitcost > worst_bitcost:
            worst_bitcost = bitcost
        total_bitcost += bitcost

        abserr = abs(float(err))
        if math.isfinite(abserr):
            if abserr > maxerr:
                maxerr = abserr
        else:
            infs += 1
    # print(f'{timeouts!s} inputs ran for more than 200 steps.')
    # print(f'worst point was {maxerr!s}.')
    return timeouts, infs, maxerr, worst_bitcost, total_bitcost

def sweep_stage(bound, expbits, res_bits, diff_bits, scale_bits):
    formatted = sqrt_core.format(
        overall_prec = '(float 8 16)',
        res_prec = f'(float {expbits!s} {res_bits!s})',
        diff_prec = f'(float {expbits!s} {diff_bits!s})',
        scale_prec = f'(float {expbits!s} {scale_bits!s})',
    )
    core = fpcparser.compile1(formatted)

    return run_and_eval_bitcost(core, bound)

    # results = run_tests(core, bound)
    # #counted, worst, avg = oob(results)
    # return oob(results)

from multiprocessing import Pool


def sweep():
    bound = 1/100
    extrabits = 2
    maxbits = 21

    cfgs = 0


    frontier = []
    frontier_bounded = []
    metrics = (operator.lt, operator.lt, operator.lt, operator.lt, operator.lt)

    with Pool() as p:
        result_buf = []

        print('building async result list')

        for expbits in range(4,9):
            for res_bits in range(expbits + extrabits, maxbits):
                for diff_bits in range(expbits + extrabits, maxbits):
                    for scale_bits in range(expbits + extrabits, maxbits):
                        args = bound, expbits, res_bits, diff_bits, scale_bits
                        result_buf.append((args, p.apply_async(sweep_stage, args)))

        print(f'waiting for {len(result_buf)!s} results to come back')

        for args, result in result_buf:
            bound,  *config = args
            expbits, res_bits, diff_bits, scale_bits =  config
            result_get = result.get()
            timeouts, infs, worst, worst_bitcost, bitcost = result_get

            cfgs += 1
            print(f' -- {cfgs!s} -- ran {config!r}, got {result_get!r}')

            if timeouts == 0 and infs == 0:
                frontier_elt = (config, result_get)

                updated, frontier = search.update_frontier(frontier, frontier_elt, metrics)
                if updated:
                    print('New (unbounded) frontier:')
                    search.print_frontier(frontier)

                if worst < bound:
                    updated, frontier_bounded = search.update_frontier(frontier_bounded, frontier_elt, metrics)
                    if updated:
                        print('New BOUNDED frontier:')
                        search.print_frontier(frontier_bounded)

    print(f'tried {cfgs!s} configs, done\n')
    print('Final frontier:')
    search.print_frontier(frontier)
    print('Final bounded frontier:')
    search.print_frontier(frontier_bounded)
    print(flush=True)

    return frontier, frontier_bounded



# there are 5 metrics:

# how many examples timed out
# how many examples reported inf
# the worst error (not residual!) of an example
# the worst individual bitcost
# the total bitcost (sum of examples)

# And here are the frontiers:

# tried 11375 configs, done

# Final frontier:
# {
#   [4, 14, 15, 8] : (0, 0, 0.010319312515875811, 2722, 583906)
#   [4, 15, 15, 8] : (0, 0, 0.009583499933366824, 2309, 586235)
#   [4, 15, 15, 9] : (0, 0, 0.009225947422650371, 2324, 589008)
#   [4, 15, 15, 10] : (0, 0, 0.009225947422650371, 1900, 592208)
#   [4, 15, 16, 9] : (0, 0, 0.008814576415149933, 2342, 592590)
#   [4, 15, 16, 10] : (0, 0, 0.008814576415149933, 1914, 596226)
#   [4, 15, 16, 11] : (0, 0, 0.008540409355627165, 1926, 599862)
#   [4, 16, 15, 10] : (0, 0, 0.008913438213184577, 1914, 595794)
# }
# Final bounded frontier:
# {
#   [4, 15, 15, 8] : (0, 0, 0.009583499933366824, 2309, 586235)
#   [4, 15, 15, 9] : (0, 0, 0.009225947422650371, 2324, 589008)
#   [4, 15, 15, 10] : (0, 0, 0.009225947422650371, 1900, 592208)
#   [4, 15, 16, 9] : (0, 0, 0.008814576415149933, 2342, 592590)
#   [4, 15, 16, 10] : (0, 0, 0.008814576415149933, 1914, 596226)
#   [4, 15, 16, 11] : (0, 0, 0.008540409355627165, 1926, 599862)
#   [4, 16, 15, 10] : (0, 0, 0.008913438213184577, 1914, 595794)
# }

# ([([4, 14, 15, 8], (0, 0, 0.010319312515875811, 2722, 583906)), ([4, 15, 15, 8], (0, 0, 0.009583499933366824, 2309, 586235)), ([4, 15, 15, 9], (0, 0, 0.009225947422650371, 2324, 589008)), ([4, 15, 15, 10], (0, 0, 0.009225947422650371, 1900, 592208)), ([4, 15, 16, 9], (0, 0, 0.008814576415149933, 2342, 592590)), ([4, 15, 16, 10], (0, 0, 0.008814576415149933, 1914, 596226)), ([4, 15, 16, 11], (0, 0, 0.008540409355627165, 1926, 599862)), ([4, 16, 15, 10], (0, 0, 0.008913438213184577, 1914, 595794))], [([4, 15, 15, 8], (0, 0, 0.009583499933366824, 2309, 586235)), ([4, 15, 15, 9], (0, 0, 0.009225947422650371, 2324, 589008)), ([4, 15, 15, 10], (0, 0, 0.009225947422650371, 1900, 592208)), ([4, 15, 16, 9], (0, 0, 0.008814576415149933, 2342, 592590)), ([4, 15, 16, 10], (0, 0, 0.008814576415149933, 1914, 596226)), ([4, 15, 16, 11], (0, 0, 0.008540409355627165, 1926, 599862)), ([4, 16, 15, 10], (0, 0, 0.008913438213184577, 1914, 595794))])
