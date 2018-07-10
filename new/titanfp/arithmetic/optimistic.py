"""Sinking-point arithmetic with "optimistic" rounding,
and meaningful exponents on zero / inf.
"""

from ..titanic import gmpmath
from ..titanic import wolfmath
from ..titanic import sinking
from ..titanic import integral
from ..fpbench import fpcast as ast

from ..titanic.ops import OP
from .evalctx import IEEECtx


USE_GMP = True
USE_MATH = False
DEFAULT_IEEE_CTX = IEEECtx(w=11, p=53) # double

def compute_with_backend(opcode, *args, prec=54):
    result = None
    if USE_MATH:
        result_math = wolfmath.compute(opcode, *args, prec=prec)
        result = result_math
    if USE_GMP:
        result_gmp = gmpmath.compute(opcode, *args, prec=prec)
        result = result_gmp
    if USE_GMP and USE_MATH:
        if not result_gmp.is_identical_to(result_math):
            print(repr(result_gmp))
            print(repr(result_math))
            print('--')
    if result is None:
        raise ValueError('no backend specified')
    return result


def round_to_sinking_ctx(x, max_p=None, min_n=None, inexact=None, ctx=DEFAULT_IEEE_CTX):
    if max_p is None:
        p = ctx.p
    else:
        p = min(max_p, ctx.p)

    if min_n is None:
        n = ctx.n
    else:
        n = max(min_n, ctx.n)

    rounded = x.round_m(max_p=p, min_n=n)

    if inexact is None:
        return rounded
    else:
        return sinking.Sink(rounded, inexact=rounded.inexact or inexact)


def arg_to_digital(x, ctx=DEFAULT_IEEE_CTX):
    result = gmpmath.mpfr_to_digital(gmpmath.mpfr(x, ctx.p + 1))
    return round_to_sinking_ctx(result, ctx=ctx)


def smallest_p(*args):
    p = None
    for x in args:
        if x.inexact and (p is None or x.p < p):
            p = x.p
    return p

def largest_n(*args):
    n = None
    for x in args:
        if x.inexact and (n is None or x.n > n):
            n = x.n
    return n

def computation_prec(p, ctx):
    # n is completely ignored: may do extra work, if we're limited by n
    if p is None:
        return max(2, ctx.p + 1)
    else:
        return max(2, min(p, ctx.p) + 1)


def add(x1, x2, ctx):
    p = None
    n = largest_n(x1, x2)
    prec = computation_prec(p, ctx)
    result = compute_with_backend(OP.add, x1, x2, prec=prec)
    inexact = x1.inexact or x2.inexact or result.inexact
    return round_to_sinking_ctx(result, max_p=p, min_n=n, inexact=inexact, ctx=ctx)

def sub(x1, x2, ctx):
    p = None
    n = largest_n(x1, x2)
    prec = computation_prec(p, ctx)
    result = compute_with_backend(OP.sub, x1, x2, prec=prec)
    inexact = x1.inexact or x2.inexact or result.inexact
    return round_to_sinking_ctx(result, max_p=p, min_n=n, inexact=inexact, ctx=ctx)

def mul(x1, x2, ctx):
    p = smallest_p(x1, x2)
    n = None
    prec = computation_prec(p, ctx)
    result = compute_with_backend(OP.mul, x1, x2, prec=prec)
    inexact = x1.inexact or x2.inexact or result.inexact
    return round_to_sinking_ctx(result, max_p=p, min_n=n, inexact=inexact, ctx=ctx)

def div(x1, x2, ctx):
    p = smallest_p(x1, x2)
    n = None
    prec = computation_prec(p, ctx)
    result = compute_with_backend(OP.div, x1, x2, prec=prec)
    inexact = x1.inexact or x2.inexact or result.inexact
    return round_to_sinking_ctx(result, max_p=p, min_n=n, inexact=inexact, ctx=ctx)

def neg(x, ctx):
    # this shouldn't actually change anything...
    p = smallest_p(x)
    n = largest_n(x)
    prec = computation_prec(p, ctx)
    result = compute_with_backend(OP.neg, x, prec=prec)
    inexact = x.inexact or result.inexact
    # we go through the same work as other computations, as negating a number
    # that doesn't fit in the context might still need to round it, etc.
    return round_to_sinking_ctx(result, max_p=p, min_n=n, inexact=inexact, ctx=ctx)
    
def sqrt(x, ctx):
    p = smallest_p(x)
    n = None
    prec = computation_prec(p, ctx)
    result = compute_with_backend(OP.sqrt, x, prec=prec)
    inexact = x.inexact or result.inexact
    return round_to_sinking_ctx(result, max_p=p, min_n=n, inexact=inexact, ctx=ctx)

def floor(x, ctx):
    if x.n >= -1:
        # floor is a no-op, so round and return
        return round_to_sinking_ctx(x, ctx=ctx)        
    
    # Note that MPFR has screwy return codes for rounding functions - it sets the RC
    # as if the operation was supposed to be the identity, and indicates a non-integer
    # was rounded with a rc value of 2 rather than 1.

    # To get around this completely, we give MPFR enough precision that it won't have
    # to round any more to represent the result, and institute our own exactness
    # behavior.

    prec = max(2, x.p)
    result = compute_with_backend(OP.floor, x, prec=prec)
    if x.is_integer() and x.inexact:
        # this rc handling is crude, as we're temporarily creating a bogus answer
        # which has the right return code but the wrong value of n, and then depending
        # on the later rounding to fix the value of n
        if result.is_zero():
            result = -abs(result)
            rc = 1
        elif result.negative:
            rc = 1
        else: # not result.negative
            rc = -1
        result = sinking.Sink(result, inexact=True, rc=rc)
    else:
        result = sinking.Sink(result, inexact=False, rc=0)

    return round_to_sinking_ctx(result, ctx=ctx)

def fmod(x1, x2, ctx):
    # x1 is the numerator, x2 is the modulus.
    # fmod computes x1 - (x2 * i) where i is an integer such that the resut is less than x2.
    # we are limited by the n of this subtraction, which is effectively the n of x1 and the
    # p of x2, because the integer from the multiply is exact and it will align the exponents.
    p = smallest_p(x2)
    n = largest_n(x1)
    prec = computation_prec(p, ctx)
    result = compute_with_backend(OP.fmod, x1, x2, prec=prec)
    inexact = x1.inexact or x2.inexact or result.inexact
    return round_to_sinking_ctx(result, max_p=p, min_n=n, inexact=inexact, ctx=ctx)

def pow(x1, x2, ctx):
    p = smallest_p(x1, x2)
    # we might lose some more bits if the exponent is far from zero
    lost_bits = abs(x2.e)
    p = max(0, p - lost_bits)
    n = None
    prec = computation_prec(p, ctx)
    result = compute_with_backend(OP.pow, x1, x2, prec=prec)
    inexact = x1.inexact or x2.inexact or result.inexact
    return round_to_sinking_ctx(result, max_p=p, min_n=n, inexact=inexact, ctx=ctx)

def sin(x, ctx):
    n = largest_n(x)
    # precision is the number of bits left from a subtraction with pi...
    if n is None:
        p = None
    else:
        p = -n
    # We don't actually want to limit n in the computation
    n = None

    prec = computation_prec(p, ctx)
    result = compute_with_backend(OP.sin, x, prec=prec)
    inexact = x.inexact or result.inexact
    return round_to_sinking_ctx(result, max_p=p, min_n=n, inexact=inexact, ctx=ctx)

def acos(x, ctx):
    print('ACOS - not limiting precision')
    p = None
    n = None
    prec = computation_prec(p, ctx)
    result = compute_with_backend(OP.sqrt, x, prec=prec)
    inexact = x.inexact or result.inexact
    return round_to_sinking_ctx(result, max_p=p, min_n=n, inexact=inexact, ctx=ctx)
    

def interpret(core, args, ctx=None):
    """FPCore interpreter for "optimistic" Titanic sinking-point."""

    if len(core.inputs) != len(args):
        raise ValueError('incorrect number of arguments: got {}, expecting {} ({})'
                         .format(len(args), len(core.inputs), ' '.join((name for name, props in core.inputs))))

    if ctx is None:
        ctx = IEEECtx(props=core.props)

    for arg, (name, props) in zip(args, core.inputs):
        if props:
            local_ctx = IEEECtx(w=ctx.w, p=ctx.p, props=props)
        else:
            local_ctx = ctx

        if isinstance(arg, sinking.Sink):
            argval = arg
        else:
            argval = arg_to_digital(arg, local_ctx)
        ctx.let([(name, argval)])

    return evaluate(core.e, ctx)


def evaluate(e, ctx):
    """Recursive expression evaluator, with much isinstance()."""

    # ValueExpr

    if isinstance(e, ast.Val):
        s = e.value.upper()
        if s == 'TRUE':
            return True
        elif s == 'FALSE':
            return False
        else:
            return arg_to_digital(e.value, ctx)

    elif isinstance(e, ast.Var):
        # never rounded
        return ctx.bindings[e.value]

    # control flow

    elif isinstance(e, ast.Ctx):
        newctx = IEEECtx(props=e.props) # TODO: inherit properties?
        newctx.let(ctx.bindings)
        return evaluate(e.body, newctx)

    elif isinstance(e, ast.If):
        if evaluate(e.cond, ctx):
            return evaluate(e.then_body, ctx)
        else:
            return evaluate(e.else_body, ctx)

    elif isinstance(e, ast.Let):
        bindings = [(name, evaluate(expr, ctx)) for name, expr in e.let_bindings]
        newctx = ctx.clone().let(bindings)
        return evaluate(e.body, newctx)

    elif isinstance(e, ast.While):
        bindings = [(name, evaluate(init_expr, ctx)) for name, init_expr, update_expr in e.while_bindings]
        newctx = ctx.clone().let(bindings)
        while evaluate(e.cond, newctx):
            bindings = [(name, evaluate(update_expr, newctx)) for name, init_expr, update_expr in e.while_bindings]
            newctx = newctx.clone().let(bindings)
        return evaluate(e.body, newctx)

    # Unary/Binary/NaryExpr

    else:
        children = [evaluate(child, ctx) for child in e.children]

        if isinstance(e, ast.Neg):
            return neg(*children, ctx)

        elif isinstance(e, ast.Sqrt):
            return sqrt(*children, ctx)

        elif isinstance(e, ast.Add):
            return add(*children, ctx)

        elif isinstance(e, ast.Sub):
            return sub(*children, ctx)

        elif isinstance(e, ast.Mul):
            return mul(*children, ctx)

        elif isinstance(e, ast.Div):
            return div(*children, ctx)

        elif isinstance(e, ast.Floor):
            return floor(*children, ctx)

        elif isinstance(e, ast.Fmod):
            return fmod(*children, ctx)

        elif isinstance(e, ast.Pow):
            return pow(*children, ctx)

        elif isinstance(e, ast.Sin):
            return sin(*children, ctx)

        elif isinstance(e, ast.Acos):
            return acos(*children, ctx)

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

fpc_loop = fpcparser.compile(
"""(FPCore ()
 (while (< x 100) ([x 0 (+ x PI)]) x))
""")[0]
