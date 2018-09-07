"""FPCore context canonicalizer and condenser."""


from ..fpbench import fpcast as ast
from . import interpreter
from . import evalctx


prop_uses_real_precision = {'pre', 'spec'}
op_is_boolean = {
    ast.LT,
    ast.GT,
    ast.LEQ,
    ast.GEQ,
    ast.EQ,
    ast.NEQ,
    ast.Isfinite,
    ast.Isinf,
    ast.Isnan,
    ast.Isnormal,
    ast.Signbit,
    ast.And,
    ast.Or,
    ast.Not,
}


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
        if len(ctx.props) == 0 or type(e) in op_is_boolean:
            return type(e)(*children)
        else:
            return ast.Ctx(props=ctx.props, body=type(e)(*children))

    # translator interface

    @classmethod
    def translate(cls, core, ctx=None,
                  propagate={'precision', 'round', 'math-library'},
                  recurse={'pre', 'spec'}):
        if ctx is None:
            ctx = cls.ctype(props={k:v for k, v in core.props.items() if k in propagate})
        else:
            ctx = ctx.let(props={k:v for k, v in core.props.items() if k in propagate})

        inputs = [(name, ctx.let(props=props).props) for name, props in core.inputs]
        e = cls.evaluate(core.e, ctx)

        props = {}
        for k, v in core.props.items():
            if k in recurse:
                if k in propagate:
                    raise ValueError('Canonicalizer: cannot propagate and recurse on the same property: {}'
                                     .format(str(k)))
                elif isinstance(v, ast.Expr):
                    if k in prop_uses_real_precision:
                        rectx = ctx.let(props={'precision': ast.Var('real')})
                    else:
                        rectx = ctx
                    print(rectx)
                    props[k] = cls.evaluate(v, rectx)
                else:
                    props[k] = v
            elif k not in propagate:
                props[k] = v

        return ast.FPCore(inputs, e, props=props)


class Condenser(interpreter.Evaluator):
    """FPCore condenser.
    Remove explicit annotations that are known to be redundant.
    This does not result in a minimal set of annotations: there
    could be places where some annotations could be pulled up
    into a parent and merged together. The Minimizer handles
    those cases.
    """

    @classmethod
    def _eval_var(cls, e, ctx):
        return e

    @classmethod
    def _eval_val(cls, e, ctx):
        return e

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
        return type(e)(*(cls.evaluate(child, ctx) for child in e.children))

    # all of the interesting work is here
    @classmethod
    def _eval_ctx(cls, e, ctx):
        interesting_props = {k:v for k, v in e.props.items() if k not in ctx.props or ctx.props[k] != v}
        if interesting_props:
            return ast.Ctx(props=interesting_props, body=cls.evaluate(e.body, ctx.let(props=e.props)))
        else:
            return cls.evaluate(e.body, ctx.let(props=e.props))

    # translator interface
    @classmethod
    def translate(cls, core, ctx=None, recurse={'pre', 'spec'}):
        if ctx is None:
            ctx = cls.ctype(props=core.props)
        else:
            ctx = ctx.let(props=core.props)

        inputs = [(name, {k:v for k, v in props.items() if k not in ctx.props or ctx.props[k] != v})
                  for name, props in core.inputs]
        e = cls.evaluate(core.e, ctx)

        props = {}
        for k, v in core.props.items():
            if k in recurse and isinstance(v, ast.Expr):
                if k in prop_uses_real_precision:
                    rectx = ctx.let(props={'precision': ast.Var('real')})
                else:
                    rectx = ctx
                props[k] = cls.evaluate(v, rectx)
            else:
                props[k] = v

        return ast.FPCore(inputs, e, props=props)


class Minimizer(interpreter.Evaluator):
    """FPCore minimizer.
    Pull all annotations up to the top level, so that each annotation
    appears in as few places and is inherited as much as possible.
    If an annotation appears on some, but not all, children of a node, it will be
    written explicitly for all of the children. Annotations could be
    minimized further by choosing the "most popular" annotation in these
    cases and carrying that one up to the parent.
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
            return child, ctx_used
        else:
            return ast.Ctx(props=annotations, body=child), ctx_used

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
        children_used = (cls._annotate(child, ctx) for child in e.children)
        children, used = zip(*children_used)
        return type(e)(*children), ctx, any(used) or type(e) not in op_is_boolean

    # translator interface

    @classmethod
    def translate(cls, core, ctx=None, recurse={'pre', 'spec'}):
        if ctx is None:
            ctx = cls.ctype(props=core.props)
        else:
            ctx = ctx.let(props=core.props)

        e, ctx, ctx_used = cls.evaluate(core.e, ctx)

        inputs = []
        for name, props in core.inputs:
            annotations = {}
            for k, v in props.items():
                if k not in ctx.props or v != ctx.props[k]:
                    annotations[k] = v
            inputs.append((name, annotations))

        reprops = {}
        for prop in recurse:
            if prop in ctx.props and isinstance(ctx.props[prop], ast.Expr):
                if prop in prop_uses_real_precision:
                    local_ctx = ctx.let(props={'precision': ast.Var('real')})
                else:
                    local_ctx = ctx

                re, rectx, rectx_used = cls.evaluate(ctx.props[prop], local_ctx)

                if rectx_used:
                    annotations = {}
                    for k, v in rectx.props.items():
                        if prop in prop_uses_real_precision and k == 'precision':
                            if str(v) != 'real':
                                annotations[k] = v
                        elif k not in ctx.props or v != ctx.props[k]:
                            annotations[k] = v
                    if annotations:
                        reprops[prop] = ast.Ctx(props=annotations, body=re)
                    else:
                        reprops[prop] = re
                else:
                    reprops[prop] = re

        # technically, we should run a full merge on all of the inputs and the
        # recurse props, but eh

        return ast.FPCore(inputs, e, props=ctx.let(props=reprops).props)
