from .fpbench import fpcparser
from .arithmetic import ieee754, optimistic, np, evalctx
from .titanic import sinking

from .arithmetic import core2math
from .titanic import wolfmath

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
    (* 2 3.14159))
)
""")[0]

fpc_sinpow = fpcparser.compile(
    """(FPCore (x) (sin (pow 2 x)))
""")[0]

floatctx = evalctx.EvalCtx(props={'precision':'binary32'})
doublectx = evalctx.EvalCtx(props={'precision':'binary64'})
bigctx = evalctx.EvalCtx(w = 20, p = 16360)

repl = wolfmath.MathRepl()

def compare(core, *inputs, ctx=None):
    result_ieee = ieee754.interpret(core, inputs, ctx).collapse()
    result_sink = optimistic.interpret(core, inputs, ctx)
    result_np = np.interpret(core, inputs, ctx)
    print(result_ieee, result_sink, result_np)
    print(repr(result_sink))
    print(result_ieee == sinking.Sink(result_np))

    math_args, math_expr = core2math.compile(core)

    math_inputs = []
    for arg, name in zip(inputs, math_args):
        # TODO always uses double, hmmm
        value = sinking.Sink(arg)
        math_inputs.append([name, value.to_math()])

    core_expr = 'With[{{{}}}, {}]'.format(', '.join((name + ' = ' + value for name, value in math_inputs)), math_expr)

    # TODO also always double
    result_exact = repl.evaluate_to_sink(core_expr)
    print(result_exact)
    print('')


import numpy

def floats_near(s, n):
    f = sinking.Sink(numpy.float32(s))
    nearby = []

    for i in range(n):
        nearby.append(f)
        f = f.away()

    return nearby

def get_vertices(f):
    result_ieee = sinking.Sink(np.interpret(fpc_sinpow, [f.to_float(numpy.float32)], floatctx))
    result_double = ieee754.interpret(fpc_sinpow, [f], doublectx)
    result_sink = optimistic.interpret(fpc_sinpow, [f], floatctx)
    print(result_ieee, result_double, result_sink)
    narrowed = result_sink.narrow(n=result_sink.n - 1)
    return ['Point[{{{}, {}}}, VertexColors -> {{{}}}]'.format(f.to_math(), y.to_math(), color)
            for y, color in [[result_ieee, 'Black'], [narrowed.above(), 'Red'], [narrowed.below(), 'Red']]]

def make_plot(s, n):
    fs = floats_near(s, n)
    points = []
    for f in fs:
        points += get_vertices(f)

    plot_cmd = 'Export["test.pdf", Plot[Sin[2^x], {{x, {}, {}}}, PlotLabel -> "sin(2^x)", Epilog -> {{PointSize[0.02], {}}}], ImageResolution -> 1200]'.format(
        fs[0].to_math(),
        fs[-1].to_math(),
        ', '.join(p for p in points))

    print(plot_cmd)
    print(repl.run(plot_cmd))
