"""Emulated IEEE 754 floating-point arithmetic.
"""

from ..titanic import gmpmath
from ..titanic import wolfmath
from ..titanic import sinking
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


def round_to_ieee_ctx(x, inexact=None, ctx=DEFAULT_IEEE_CTX):
    rounded = x.round_m(max_p=ctx.p, min_n=ctx.n)
    if inexact is None:
        return rounded
    else:
        return sinking.Sink(rounded, inexact=rounded.inexact or inexact)


def arg_to_digital(x, ctx=DEFAULT_IEEE_CTX):
    result = gmpmath.mpfr_to_digital(gmpmath.mpfr(x, ctx.p + 1))
    return round_to_ieee_ctx(result, inexact=None, ctx=ctx)


def add(x1, x2, ctx):
    prec = max(2, ctx.p + 1)
    result = compute_with_backend(OP.add, x1, x2, prec=prec)
    inexact = x1.inexact or x2.inexact or result.inexact
    return round_to_ieee_ctx(result, inexact, ctx)

def sub(x1, x2, ctx):
    prec = max(2, ctx.p + 1)
    result = compute_with_backend(OP.sub, x1, x2, prec=prec)
    inexact = x1.inexact or x2.inexact or result.inexact
    return round_to_ieee_ctx(result, inexact, ctx)

def mul(x1, x2, ctx):
    prec = max(2, ctx.p + 1)
    result = compute_with_backend(OP.mul, x1, x2, prec=prec)
    inexact = x1.inexact or x2.inexact or result.inexact
    return round_to_ieee_ctx(result, inexact, ctx)

def div(x1, x2, ctx):
    prec = max(2, ctx.p + 1)
    result = compute_with_backend(OP.div, x1, x2, prec=prec)
    inexact = x1.inexact or x2.inexact or result.inexact
    return round_to_ieee_ctx(result, inexact, ctx)

def neg(x, ctx):
    prec = max(2, ctx.p + 1)
    result = compute_with_backend(OP.neg, x, prec=prec)
    inexact = x.inexact or result.inexact
    return round_to_ieee_ctx(result, inexact, ctx)

def sqrt(x, ctx):
    prec = max(2, ctx.p + 1)
    result = compute_with_backend(OP.sqrt, x, prec=prec)
    inexact = x.inexact or result.inexact
    return round_to_ieee_ctx(result, inexact, ctx)

def floor(x, ctx):
    prec = max(2, ctx.p + 1)
    result = compute_with_backend(OP.floor, x, prec=prec)
    inexact = (x.inexact or result.inexact) and result.n > -1 # TODO: correct?
    return round_to_ieee_ctx(result, inexact, ctx)

def fmod(x1, x2, ctx):
    prec = max(2, ctx.p + 1)
    result = compute_with_backend(OP.fmod, x1, x2, prec=prec)
    inexact = x1.inexact or x2.inexact or result.inexact
    return round_to_ieee_ctx(result, inexact, ctx)

def pow(x1, x2, ctx):
    prec = max(2, ctx.p + 1)
    result = compute_with_backend(OP.pow, x1, x2, prec=prec)
    inexact = x1.inexact or x2.inexact or result.inexact
    return round_to_ieee_ctx(result, inexact, ctx)

def sin(x, ctx):
    prec = max(2, ctx.p + 1)
    result = compute_with_backend(OP.sin, x, prec=prec)
    inexact = x.inexact or result.inexact
    return round_to_ieee_ctx(result, inexact, ctx)

def acos(x, ctx):
    prec = max(2, ctx.p + 1)
    result = compute_with_backend(OP.acos, x, prec=prec)
    inexact = x.inexact or result.inexact
    return round_to_ieee_ctx(result, inexact, ctx)


def interpret(core, args, ctx=None):
    """FPCore interpreter for IEEE 754-like arithmetic."""

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


def interpret_pre(core, args, ctx=None):
    if core.pre is None:
        raise ValueError('core has no preconditions')
    
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

    return evaluate(core.pre, ctx)


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

    # and Digits

    elif isinstance(e, ast.Digits):
        raise ValueError('unimplemented')
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