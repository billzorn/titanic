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


def sweep_stage(bound, expbits, res_bits, diff_bits, scale_bits):
    formatted = sqrt_core.format(
        overall_prec = '(float 8 16)',
        res_prec = f'(float {expbits!s} {res_bits!s})',
        diff_prec = f'(float {expbits!s} {diff_bits!s})',
        scale_prec = f'(float {expbits!s} {scale_bits!s})',
    )
    core = fpcparser.compile1(formatted)
    results = run_tests(core, bound)
    #counted, worst, avg = oob(results)
    return oob(results)

from multiprocessing import Pool

def sweep():
    bound = 1/100
    cheapest = 1000
    badness = 1000
    cheapest_config = None

    cfgs = 0

    with Pool() as p:
        result_buf = []

        print('building async result list')

        for expbits in range(3,9):
            for res_bits in range(expbits + 1, 18):
                for diff_bits in range(expbits + 1, 18):
                    for scale_bits in range(expbits + 1, 18):
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

        print('waiting for results to come back')

        for args, result in result_buf:
            bound, expbits, res_bits, diff_bits, scale_bits = args
            counted, worst, avg = result.get()

            cfgs += 1
            print(f'ran {(expbits, res_bits, diff_bits, scale_bits)!r}, got {(counted, worst, avg)!r}')

            if counted == 0 and worst < bound:
                cost = res_bits + diff_bits + scale_bits
                if cost <= cheapest:
                    if worst < badness:
                        cheapest = cost
                        badness = worst
                        cheapest_config = (expbits, res_bits, diff_bits, scale_bits)
                        print(f'  NEW best cost: {cheapest!r} w/ {cheapest_config!r}')

    return cheapest, cheapest_config


# we got an answer:

# (33, (3, 11, 12, 10))
# >>> sweep_stage(1/100, 3, 11, 12, 10)
# (0, 0.00894894614631192, 0.002602821809623318)
