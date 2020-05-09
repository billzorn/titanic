import math

from .fpbench import fpcparser
from .arithmetic import mpmf, ieee754, evalctx, analysis
from .arithmetic.mpmf import Interpreter

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
    total_bitcost = 0
    for arg, result, err, steps, bitcost in results:
        if steps > 200:
            timeouts += 1
        total_bitcost += bitcost

        abserr = abs(float(err))
        if math.isfinite(abserr):
            if abserr > maxerr:
                maxerr = abserr
        else:
            infs += 1
    # print(f'{timeouts!s} inputs ran for more than 200 steps.')
    # print(f'worst point was {maxerr!s}.')
    return timeouts, infs, maxerr, total_bitcost

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
    cheapest = float('inf')
    badness = float('inf')
    cheapest_config = None

    cfgs = 0

    with Pool() as p:
        result_buf = []

        print('building async result list')

        for expbits in range(2,9):
            for res_bits in range(expbits + 1, 19):
                for diff_bits in range(expbits + 1, 19):
                    for scale_bits in range(expbits + 1, 19):
                        args = bound, expbits, res_bits, diff_bits, scale_bits
                        result_buf.append((args, p.apply_async(sweep_stage, args)))

                        # formatted = sqrt_core.format(
                        #      overall_prec = '(float 8 16)',
                        #     res_prec = f'(float {expbits!s} {res_bits!s})',
                        #     diff_prec = f'(float {expbits!s} {diff_bits!s})',
                        #     scale_prec = f'(float {expbits!s} {scale_bits!s})',
                        # )
                        # core = fpcparser.compile1(formatted)
                        # results = run_tests(core, bound)
                        # counted, worst, avg = oob(results)

        print(f'waiting for {len(result_buf)!s} results to come back')

        for args, result in result_buf:
            bound,  *config = args
            expbits, res_bits, diff_bits, scale_bits =  config
            result_get = result.get()
            timeouts, infs, worst, bitcost = result_get

            cfgs += 1
            print(f'ran {config!r}, got {result_get!r}')

            if timeouts == 0 and infs == 0 and worst < bound:
                cost = bitcost
                if cost <= cheapest:
                    if worst < badness:
                        cheapest = cost
                        badness = worst
                        cheapest_config = config
                        print(f'  NEW best cost: {cheapest!r}, badness {badness!r}, w/ {cheapest_config!r}')

        print(f'tried {cfgs!s} configs, done')

    return cheapest, badness, cheapest_config


# got an answer:

# (592590, 0.008814576415149933, [4, 15, 16, 9])
# >>> sweep_stage(1/100, 4, 15, 16, 9)
# (0, 0, 0.008814576415149933, 592590)
