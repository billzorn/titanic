"""Sinking-point arithmetic with "optimistic" rounding,
and meaningful exponents on zero / inf.
"""

import gmpy2 as gmp

from ..titanic import gmpmath
from ..titanic import sinking
from ..fpbench import fpcast as ast


class EvalCtx(object):
    def __init__(self, w=11, p=53):
        # IEEE 754-like
        self.w = w
        self.p = p
        self.emax = (1 << (self.w - 1)) - 1
        self.emin = 1 - self.emax
        self.n = self.emin - self.p
        # variables and stuff
        self.bindings = {}


def interpret(core, args, ctx):
    """Main FPCore interpreter."""

    varnames = core.inputs
    if len(varnames) != len(args):
        raise ValueError('incorrect number of arguments: got {}, expecting {} ({})'
                         .format(len(args), len(varnames), ' '.join(varnames)))

    ctx.bindings.update(zip(varnames, [sinking.Sink(arg, max_p=ctx.p, min_n=ctx.n) for arg in args]))

    return evaluate(core.e, ctx)


def evaluate(e, ctx):
    """Recursive expression evaluator, with much isinstance()."""

    # ValueExpr

    if isinstance(e, ast.Val):
        # TODO precision
        return sinking.Sink(e.value, max_p=ctx.p, min_n=ctx.n)

    elif isinstance(e, ast.Var):
        return ctx.bindings[e.value]

    # Unary/Binary/NaryExpr

    else:
        children = [evaluate(child, ctx) for child in e.children]

        if isinstance(e, ast.Neg):
            # always exact
            return -children[0]

        elif isinstance(e, ast.Sqrt):
            n = ctx.n
            p = min(child._p if child._inexact else ctx.p for child in children)
            # p = min(children[0]._p, ctx.p) if children[0]._inexact else ctx.p
            return gmpmath.withnprec(gmp.sqrt, *children, min_n=n, max_p=p)

        elif isinstance(e, ast.Add):
            n = max(child._n if child._inexact else ctx.n for child in children)
            p = ctx.p
            return gmpmath.withnprec(gmp.add, *children, min_n=n, max_p=p)

        elif isinstance(e, ast.Sub):
            n = max(child._n if child._inexact else ctx.n for child in children)
            p = ctx.p
            return gmpmath.withnprec(gmp.sub, *children, min_n=n, max_p=p)

        elif isinstance(e, ast.Mul):
            n = ctx.n
            p = min(child._p if child._inexact else ctx.p for child in children)
            return gmpmath.withnprec(gmp.mul, *children, min_n=n, max_p=p)

        elif isinstance(e, ast.Div):
            n = ctx.n
            p = min(child._p if child._inexact else ctx.p for child in children)
            return gmpmath.withnprec(gmp.div, *children, min_n=n, max_p=p)

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
