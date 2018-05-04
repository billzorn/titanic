from .fpbench import fpcparser
from .arithmetic import ieee754, optimistic, evalctx

fpc_minimal = fpcparser.compile(
"""(FPCore (a b) (- (+ a b) a))
""")[0]

fpc_example = fpcparser.compile(
"""(FPCore (a b c)
 :name "NMSE p42, positive"
 :cite (hamming-1987 herbie-2015)
 :fpbench-domain textbook
 :pre (and (>= (* b b) (* 4 (* a c))) (!= a 0))
 (/ (+ (- b) (sqrt (- (* b b) (* 4 (* a c))))) (* 2 a)))
""")[0]

fpc_fmod2pi = fpcparser.compile(
"""(FPCore ()
 (- (* 2 (+ (+ (* 4 7.8539812564849853515625e-01) (* 4 3.7748947079307981766760e-08)) (* 4 2.6951514290790594840552e-15)))
    (* 2 PI))
)
""")[0]

fpc_sinpow = fpcparser.compile(
"""(FPCore (x) (sin (pow 2 x)))
""")[0]

floatctx = evalctx.EvalCtx(props={'precision':'binary32'})
doublectx = evalctx.EvalCtx(props={'precision':'binary64'})
bigctx = evalctx.EvalCtx(w = 20, p = 16360)

def compare(core, *inputs, ctx=None):
    result_ieee = ieee754.interpret(core, inputs, ctx).collapse()
    result_sink = optimistic.interpret(core, inputs, ctx)
    print(result_ieee, result_sink)
