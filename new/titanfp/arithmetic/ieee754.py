"""Sinking-point arithmetic with "optimistic" rounding,
and meaningful exponents on zero / inf.
"""

import gmpy2 as gmp

from ..titanic import gmpmath
from ..titanic import sinking
from ..fpbench import fpcast as ast

from .evalctx import EvalCtx


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

    # Handle annotations for precision-specific computations.
    if e.props:
        local_ctx = EvalCtx(w=ctx.w, p=ctx.p, props=e.props)
    else:
        local_ctx = ctx

    # ValueExpr

    if isinstance(e, ast.Val):
        # TODO precision
        return sinking.Sink(e.value, min_n=local_ctx.n, max_p=local_ctx.p)

    elif isinstance(e, ast.Var):
        # TODO better rounding and stuff
        value = ctx.bindings[e.value]
        if local_ctx is ctx:
            return value
        else:
            return value.ieee_754(local_ctx.w, local_ctx.p)

    # and Digits

    elif isinstance(e, ast.Digits):
        # TODO yolo
        spare_bits = 16
        base = sinking.Sink(e.b)
        exponent = sinking.Sink(e.e)
        scale = gmpmath.pow(base, exponent,
                            min_n = -(base.bit_length() * exponent) - spare_bits,
                            max_p = local_ctx.w + local_ctx.p + spare_bits)
        significand = sinking.Sink(e.m,
                                   min_n = local_ctx.n - spare_bits,
                                   max_p = local_ctx.p + spare_bits)
        return gmpmath.mul(significand, scale, min_n=local_ctx.n, max_p=local_ctx.p)

    # control flow

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
        n = local_ctx.n
        p = local_ctx.p

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
