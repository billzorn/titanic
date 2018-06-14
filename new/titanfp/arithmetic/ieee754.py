"""Emulated IEEE 754 floating-point arithmetic.
"""

import gmpy2 as gmp

from ..titanic import gmpmath
from ..titanic import wolfmath
from ..titanic import sinking
from ..titanic import ops
from ..fpbench import fpcast as ast
from .evalctx import EvalCtx

USE_GMP = True
USE_MATH = True

def add(x1, x2, ctx):
    prec = max(2, ctx.p + 1)
    if USE_GMP:
        result_gmp = gmpmath.compute(ops.OP.add, x1, x2, prec=prec)
        result = result_gmp
    if USE_MATH:
        result_math = wolfmath.compute(ops.OP.add, x1, x2, prec=prec)
        result = result_math
    if USE_GMP and USE_MATH:
        if not result_gmp == result_math:
            print(result_gmp)
            print(result_math)
            print('--')
    if result is None:
        raise ValueError('no backend specified')
    inexact = x1.inexact or x2.inexact or result.inexact
    return sinking.Sink(result.round_m(max_p=ctx.p, min_n=ctx.n), inexact=inexact)


def interpret(core, args, ctx=None):
    """FPCore interpreter for IEEE 754-like arithmetic."""

    if len(core.inputs) != len(args):
        raise ValueError('incorrect number of arguments: got {}, expecting {} ({})'
                         .format(len(args), len(core.inputs), ' '.join((name for name, props in core.inputs))))

    if ctx is None:
        ctx = EvalCtx(props=core.props)

    for arg, (name, props) in zip(args, core.inputs):
        if props:
            local_ctx = EvalCtx(w=ctx.w, p=ctx.p, props=props)
        else:
            local_ctx = ctx

        value = sinking.Sink(arg, min_n=local_ctx.n, max_p=local_ctx.p)
        ctx.let([(name, value)])

    return evaluate(core.e, ctx)


def evaluate(e, ctx):
    """Recursive expression evaluator, with much isinstance()."""

    # ValueExpr

    if isinstance(e, ast.Val):
        # TODO precision
        return sinking.Sink(e.value, min_n=ctx.n, max_p=ctx.p)

    elif isinstance(e, ast.Var):
        # TODO better rounding and stuff
        value = ctx.bindings[e.value]
        if ctx is ctx:
            return value
        else:
            return value.ieee_754(ctx.w, ctx.p)

    # and Digits

    elif isinstance(e, ast.Digits):
        # TODO yolo
        spare_bits = 16
        base = sinking.Sink(e.b)
        exponent = sinking.Sink(e.e)
        scale = gmpmath.pow(base, exponent,
                            min_n = -(base.bit_length() * exponent) - spare_bits,
                            max_p = ctx.w + ctx.p + spare_bits)
        significand = sinking.Sink(e.m,
                                   min_n = ctx.n - spare_bits,
                                   max_p = ctx.p + spare_bits)
        return gmpmath.mul(significand, scale, min_n=ctx.n, max_p=ctx.p)

    # control flow

    elif isinstance(e, ast.Ctx):
        return evaluate(e.body, EvalCtx(props=e.props))

    elif isinstance(e, ast.If):
        if evaluate(e.cond, ctx):
            return evaluate(e.then_body, ctx)
        else:
            return evaluate(e.else_body, ctx)

    elif isinstance(e, ast.Let):
        # somebody has to clone the context, to prevent let bindings in the subexpressions
        # from contaminating each other or the result
        bindings = [(name, evaluate(expr, ctx.clone())) for name, expr in e.let_bindings]
        ctx.let(bindings)
        return evaluate(e.body, ctx)

    # Unary/Binary/NaryExpr

    else:
        children = [evaluate(child, ctx) for child in e.children]
        n = ctx.n
        p = ctx.p

        if isinstance(e, ast.Neg):
            # always exact
            return -children[0]

        elif isinstance(e, ast.Sqrt):
            return gmpmath.sqrt(*children, min_n=n, max_p=p)

        elif isinstance(e, ast.Add):
            return gmpmath.add(*children, min_n=n, max_p=p)

        elif isinstance(e, ast.Sub):
            return gmpmath.sub(*children, min_n=n, max_p=p)

        elif isinstance(e, ast.Mul):
            return gmpmath.mul(*children, min_n=n, max_p=p)

        elif isinstance(e, ast.Div):
            return gmpmath.div(*children, min_n=n, max_p=p)

        elif isinstance(e, ast.Floor):
            return gmpmath.floor(*children, min_n=n, max_p=p)

        elif isinstance(e, ast.Fmod):
            return gmpmath.fmod(*children, min_n=n, max_p=p)

        elif isinstance(e, ast.Pow):
            return gmpmath.pow(*children, min_n=n, max_p=p)

        elif isinstance(e, ast.Sin):
            return gmpmath.sin(*children, min_n=n, max_p=p)

        elif isinstance(e, ast.LT):
            for x, y in zip(children, children[1:]):
                if not x < y:
                    return False
            return True

        elif isinstance(e, ast.GT):
            for x, y in zip(children, children[1:]):
                if not x > y:
                    return False
            return True

        elif isinstance(e, ast.LEQ):
            for x, y in zip(children, children[1:]):
                if not x <= y:
                    return False
            return True

        elif isinstance(e, ast.GEQ):
            for x, y in zip(children, children[1:]):
                if not x >= y:
                    return False
            return True

        elif isinstance(e, ast.EQ):
            for x in children:
                for y in children[1:]:
                    if not x == y:
                        return False
            return True

        elif isinstance(e, ast.NEQ):
            for x in children:
                for y in children[1:]:
                    if not x != y:
                        return False
            return True

        elif isinstance(e, ast.Expr):
            raise ValueError('unimplemented: {}'.format(repr(e)))

        else:
            raise ValueError('what is this: {}'.format(repr(e)))


from ..fpbench import fpcparser

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
