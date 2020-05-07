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
    for arg, result, err, steps in results:
        if steps > 200:
            counted += 1
        abserr = abs(float(err))
        if abserr > 0.01:
            print(arg, result, err, steps)
        if abserr > maxerr:
            maxerr = abserr
    print(f'{counted!s} inputs ran for more than 200 steps.')
    print(f'worst point was {maxerr!s}.')
    return counted
            
