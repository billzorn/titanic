import math

from .fpbench import fpcparser
from .arithmetic import mpmf, ieee754, evalctx, analysis
from .arithmetic.mpmf import Interpreter

sqrt_core = '''(FPCore sqrt_bfloat_limit (a residual_bound)
 :precision {overall_prec}

 (while* (and (< steps (# 50)) (>= (fabs residual) residual_bound))
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
    logsumerr = 0
    for arg, result, err, steps in results:
        if steps > 200:
            counted += 1
        abserr = abs(float(err))
        if abserr > 0:
            logsumerr += math.log(abserr)
        # if abserr > 0.01:
        #     print(arg, result, err, steps)
        if abserr > maxerr:
            maxerr = abserr
    # print(f'{counted!s} inputs ran for more than 200 steps.')
    # print(f'worst point was {maxerr!s}.')
    return counted, maxerr, math.exp(logsumerr / len(results))
            
def sweep():
    bound = 1/100
    cheapest = 1000
    badness = 1000
    cheapest_config = None

    cfgs = 0

    for expbits in range(2,9):
        for res_bits in range(expbits + 1, 20):
            for diff_bits in range(expbits + 1, 20):
                for scale_bits in range(expbits + 1, 20):
                    formatted = sqrt_core.format(
                         overall_prec = '(float 8 16)',
                        res_prec = f'(float {expbits!s} {res_bits!s})',
                        diff_prec = f'(float {expbits!s} {diff_bits!s})',
                        scale_prec = f'(float {expbits!s} {scale_bits!s})',
                    )
                    core = fpcparser.compile1(formatted)
                    results = run_tests(core, bound)
                    counted, worst, avg = oob(results)

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
                                              
