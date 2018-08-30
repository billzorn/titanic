"""FPCore context canonicalizer and condenser."""

from ..fpbench import fpcast as ast
from . import interpreter
from . import evalctx


class Canonicalizer(interpreter.Evaluator):
    """FPCore canonicalizer.
    Push all annotations out to the leaves, so each operation / constants
    is annotated with its full rounding context.
    """

    @classmethod
    def _eval_var(cls, e, ctx):
        return e

    @classmethod
    def _eval_val(cls, e, ctx):
        if len(ctx.props) == 0:
            return e
        else:
            return ast.Ctx(props=ctx.props, body=e)

    @classmethod
    def _eval_if(cls, e, ctx):
        return ast.If(
            cls.evaluate(e.cond, ctx),
            cls.evaluate(e.then_body, ctx),
            cls.evaluate(e.else_body, ctx),
        )

    @classmethod
    def _eval_let(cls, e, ctx):
        return ast.Let(
            [(name, cls.evaluate(expr, ctx)) for name, expr in e.let_bindings],
            cls.evaluate(e.body, ctx),
        )

    @classmethod
    def _eval_while(cls, e, ctx):
        return ast.While(
            cls.evaluate(e.cond, ctx),
            [(name, cls.evaluate(init_expr, ctx), cls.evaluate(init_expr, ctx))
             for name, init_expr, update_expr in e.while_bindings],
            cls.evaluate(e.body, ctx),
        )

    @classmethod
    def _eval_op(cls, e, ctx):
        children = [cls.evaluate(child, ctx) for child in e.children]
        if len(ctx.props) == 0:
            return type(e)(*children)
        else:
            return ast.Ctx(props=ctx.props, body=type(e)(*children))


    # translator interface

    @classmethod
    def translate(cls, core, ctx=None):
        if ctx is None:
            ctx = cls.ctype(props=core.props)
        else:
            ctx = ctx.let(props=core.props)

        inputs = [(name, ctx.let(props=props).props) for name, props in core.inputs]
        e = cls.evaluate(core.e, ctx)
        return ast.FPCore(inputs, e)
