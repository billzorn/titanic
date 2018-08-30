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
        children = (cls.evaluate(child, ctx) for child in e.children)
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


class Condenser(interpreter.Evaluator):
    """FPCore condenser.
    Pull all annotations up to the top level, so that each annotation
    appears in as few places and is inherited as much as possible.
    """

    @classmethod
    def _annotate(cls, e, ctx):
        """Given some subexpression e and its surrounding context, determine
        which properties on e are different from the surround context and
        thus need to be annotated specifically.
        """
        child, childctx = cls.evaluate(e, ctx)
        annotations = {}
        for prop in childctx.props:
            if prop not in ctx.props or childctx.props[prop] != ctx.props[prop]:
                annotations[prop] = childctx.props[prop]

        print('annotated: ' + str(e))
        print('  parent: ' + repr(ctx))
        print('  child:  ' + repr(childctx))

        if len(annotations) == 0 or isinstance(child, ast.Var):
            return child
        else:
            return ast.Ctx(props=annotations, body=child)

    @classmethod
    def _eval_var(cls, e, ctx):
        return e, ctx

    @classmethod
    def _eval_val(cls, e, ctx):
        # do nothing here; the parent will annotate if necessary
        return e, ctx

    @classmethod
    def _eval_if(cls, e, ctx):
        annotated = ast.If(
            cls._annotate(e.cond, ctx),
            cls._annotate(e.then_body, ctx),
            cls._annotate(e.else_body, ctx),
        )
        return annotated, ctx

    @classmethod
    def _eval_let(cls, e, ctx):
        annotated = ast.Let(
            [(name, cls._annotate(expr, ctx)) for name, expr in e.let_bindings],
            cls._annotate(e.body, ctx),
        )
        return annotated, ctx

    @classmethod
    def _eval_while(cls, e, ctx):
        annotated = ast.While(
            cls._annotate(e.cond, ctx),
            [(name, cls._annotate(init_expr, ctx), cls._annotate(init_expr, ctx))
             for name, init_expr, update_expr in e.while_bindings],
            cls._annotate(e.body, ctx),
        )
        return annotated, ctx

    @classmethod
    def _eval_op(cls, e, ctx):
        children = (cls._annotate(child, ctx) for child in e.children)
        return type(e)(*children), ctx

    # translator interface

    @classmethod
    def translate(cls, core, ctx=None):
        if ctx is None:
            ctx = cls.ctype(props=core.props)
        else:
            ctx = ctx.let(props=core.props)

        e, ctx = cls.evaluate(core.e, ctx)

        inputs = []
        for name, props in core.inputs:
            annotations = {}
            for prop in props:
                if prop not in ctx.props or props[prop] != ctx.props[prop]:
                    annotations[prop] = props[prop]
            inputs.append((name, annotations))

        return ast.FPCore(inputs, e, props=ctx.props)
