"""Convert FPCores into Wolfram Mathematica expressions.
"""

import re

import numpy as np

from ..titanic import sinking
from ..fpbench import fpcast as ast
from .evalctx import EvalCtx


_bad_chars = re.compile(r'[^a-z0-9]')
_starts_with_number = re.compile(r'^[0-9]')


def compile(core, ctx=None):
    """FPCore interpreter for IEEE 754-like arithmetic."""

    if ctx is None:
        ctx = EvalCtx(props=core.props)

    id_box = [1]
    already_mangled = {}

    # TODO put this in the context???
    def mangle_name(name):
        if name in already_mangled:
            return already_mangled[name]

        mangled = _bad_chars.sub('', name.lower())
        while _starts_with_number.match(mangled):
            mangled = mangled[1:]

        if mangled != name:
            var_id = id_box[0]
            mangled += 'VAR{:d}'.format(var_id)
            id_box[0] = var_id + 1

        already_mangled[name] = mangled
        return mangled

    return translate(core.e, ctx, mangle_name)


def translate(e, ctx, mangle_name):
    """Recursive expression evaluator, with much isinstance()."""

    # Handle annotations for precision-specific computations.
    if e.props:
        local_ctx = EvalCtx(w=ctx.w, p=ctx.p, props=e.props)
    else:
        local_ctx = ctx

    # ValueExpr

    if isinstance(e, ast.Val):
        # this always rounds to some FP format, which may not be what you want
        # TODO
        return sinking.Sink(e.value, min_n=local_ctx.n, max_p=local_ctx.p).to_math()

    elif isinstance(e, ast.Var):
        return mangle_name(e.value)

    # and Digits

    elif isinstance(e, ast.Digits):
        return '{:d} * {:d}^{:d}'.format(e.m, e.b, e.e)

    # control flow

    elif isinstance(e, ast.If):
        return 'If[{}, {}, {}]'.format(
            translate(e.cond, ctx, mangle_name),
            translate(e.then_body, ctx, mangle_name),
            translate(e.else_body, ctx, mangle_name),
        )

    elif isinstance(e, ast.Let):
        bindings = ['{} = {}'.format(mangle_name(name), translate(expr, ctx, mangle_name)) for name, expr in e.let_bindings]
        return 'With[{{{}}}, {}]'.format(', '.join(bindings), translate(e.body, ctx, mangle_name))

    # Unary/Binary/NaryExpr

    else:
        children = [translate(child, ctx, mangle_name) for child in e.children]

        if isinstance(e, ast.Neg):
            return '-({})'.format(*children)

        elif isinstance(e, ast.Sqrt):
            return 'Sqrt[{}]'.format(*children)

        elif isinstance(e, ast.Add):
            return '({}) + ({})'.format(*children)

        elif isinstance(e, ast.Sub):
            return '({}) - ({})'.format(*children)

        elif isinstance(e, ast.Mul):
            return '({}) * ({})'.format(*children)

        elif isinstance(e, ast.Div):
            return '({}) / ({})'.format(*children)

        elif isinstance(e, ast.Floor):
            return 'Floor[{}]'.format(*children)

        elif isinstance(e, ast.Fmod):
            return 'Mod[{}, {}]'.format(*children)

        elif isinstance(e, ast.Pow):
            return '({}) ^ ({})'.format(*children)

        elif isinstance(e, ast.Sin):
            return 'Sin[{}]'.format(*children)

        elif isinstance(e, ast.LT):
            return ' < '.join(children)

        elif isinstance(e, ast.GT):
            return ' > '.join(children)

        elif isinstance(e, ast.LEQ):
            return ' <= '.join(children)

        elif isinstance(e, ast.GEQ):
            return ' >= '.join(children)

        elif isinstance(e, ast.EQ):
            return ' == '.join(children)

        elif isinstance(e, ast.NEQ):
            return ' != '.join(children)

        elif isinstance(e, ast.Expr):
            raise ValueError('unimplemented: {}'.format(repr(e)))

        else:
            raise ValueError('what is this: {}'.format(repr(e)))
