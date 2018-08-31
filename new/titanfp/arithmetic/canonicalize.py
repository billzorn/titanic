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
            [(name, cls.evaluate(init_expr, ctx), cls.evaluate(update_expr, ctx))
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
        child, childctx, ctx_used = cls.evaluate(e, ctx)
        annotations = {}

        for prop in childctx.props:
            if prop not in ctx.props or childctx.props[prop] != ctx.props[prop]:
                annotations[prop] = childctx.props[prop]

        if len(annotations) == 0 or not ctx_used:
            return child
        else:
            return ast.Ctx(props=annotations, body=child)

    @classmethod
    def _merge_contexts(cls, children, ctx):
        children_ctxs = [cls.evaluate(child, ctx) for child in children]
        any_child_used = False
        shared = {}
        shared_props = {}
        annotated = []

        for child, childctx, child_used in children_ctxs:
            any_child_used = any_child_used or child_used
            annotations = {}
            for prop in childctx.props:
                if prop not in ctx.props or childctx.props[prop] != ctx.props[prop]:
                    # property differs from surrounding context
                    # check if it's the same on all children
                    if prop in shared:
                        is_shared = shared[prop]
                    else:
                        is_shared = True
                        for other, otherctx, other_used in children_ctxs:
                            if (other is not child and
                                other_used and
                                (prop not in otherctx.props or childctx.props[prop] != otherctx.props[prop])):
                                is_shared = False
                        shared[prop] = is_shared
                        # remember the shared property and its value
                        shared_props[prop] = childctx.props[prop]
                    # property is not shared, so we need to to annotate
                    if not is_shared:
                        annotations[prop] = childctx.props[prop]
            if len(annotations) == 0 or not child_used:
                annotated.append(child)
            else:
                annotated.append(ast.Ctx(props=annotations, body=child))

        return annotated, ctx.let(props=shared_props), any_child_used

    @classmethod
    def _eval_var(cls, e, ctx):
        return e, ctx, False

    @classmethod
    def _eval_val(cls, e, ctx):
        # do nothing here; the parent will annotate if necessary
        return e, ctx, True

    @classmethod
    def _eval_if(cls, e, ctx):
        (cond, let_body, else_body), ctx, ctx_used = cls._merge_contexts(
            [e.cond, e.then_body, e.else_body],
            ctx,
        )
        return ast.If(cond, let_body, else_body), ctx, ctx_used

    @classmethod
    def _eval_let(cls, e, ctx):
        names, child_exprs = zip(*e.let_bindings)
        (body, *exprs), ctx, ctx_used = cls._merge_contexts(
            [e.body, *child_exprs],
            ctx,
        )
        return ast.Let([*zip(names, exprs)], body), ctx, ctx_used

    @classmethod
    def _eval_while(cls, e, ctx):
        names, child_inits, child_updates = zip(*e.while_bindings)
        (cond, body, *exprs), ctx, ctx_used = cls._merge_contexts(
            [e.cond, e.body, *child_inits, *child_updates],
            ctx,
        )
        init_exprs = exprs[:len(child_inits)]
        update_exprs = exprs[len(child_updates):]
        return ast.While(cond, [*zip(names, init_exprs, update_exprs)], body), ctx, ctx_used

    @classmethod
    def _eval_op(cls, e, ctx):
        children = (cls._annotate(child, ctx) for child in e.children)
        return type(e)(*children), ctx, True

    # translator interface

    @classmethod
    def translate(cls, core, ctx=None):
        if ctx is None:
            ctx = cls.ctype(props=core.props)
        else:
            ctx = ctx.let(props=core.props)

        e, ctx, ctx_used = cls.evaluate(core.e, ctx)

        # technically, if the context isn't used in the body,
        # we should run another merge on the annotations of the
        # inputs...
        inputs = []
        for name, props in core.inputs:
            annotations = {}
            for prop in props:
                if prop not in ctx.props or props[prop] != ctx.props[prop]:
                    annotations[prop] = props[prop]
            inputs.append((name, annotations))

        return ast.FPCore(inputs, e, props=ctx.props)
