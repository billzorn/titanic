"""Use numpy to interpret FPCores.
"""

import numpy as np

from ..fpbench import fpcast as ast
from .evalctx import EvalCtx


def _np_typeof(ctx):
    if ctx.w == 11 and ctx.p == 53:
        return np.float64
    elif ctx.w == 8 and ctx.p == 24:
        return np.float32
    elif ctx.w == 5 and ctx.p == 11:
        return np.float16
    else:
        raise ValueError('context with w={}, p={} does not correspond to a numpy float type'
                         .format(ctx.w, ctx.p))


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

        # TODO: numpy rounding!
        ftype = _np_typeof(local_ctx)
        value = ftype(arg)
        ctx.let([(name, value)])

    return evaluate(core.e, ctx)


def evaluate(e, ctx):
    """Recursive expression evaluator, with much isinstance()."""

    # Handle annotations for precision-specific computations.
    if e.props:
        local_ctx = EvalCtx(w=ctx.w, p=ctx.p, props=e.props)
    else:
        local_ctx = ctx

    ftype = _np_typeof(local_ctx)

    # ValueExpr

    if isinstance(e, ast.Val):
        # TODO numpy rounding!
        return ftype(e.value)

    elif isinstance(e, ast.Var):
        # TODO better rounding and stuff
        return ftype(ctx.bindings[e.value])

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
        r = gmpmath.mul(significand, scale, min_n=local_ctx.n, max_p=local_ctx.p)
        return r.to_float(ftype)

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
            return np.sqrt(children[0])

        elif isinstance(e, ast.Add):
            return np.add(*children)

        elif isinstance(e, ast.Sub):
            return children[0] - children[1]

        elif isinstance(e, ast.Mul):
            return children[0] * children[1]

        elif isinstance(e, ast.Div):
            return children[0] / children[1]

        elif isinstance(e, ast.Floor):
            return np.floor(*children)

        elif isinstance(e, ast.Fmod):
            return np.fmod(*children)

        elif isinstance(e, ast.Pow):
            return children[0] ** children[1]

        elif isinstance(e, ast.Sin):
            return np.sin(*children)

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
