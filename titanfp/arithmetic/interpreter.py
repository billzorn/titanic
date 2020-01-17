"""Base FPCore interpreter."""


import itertools

from ..titanic import utils, digital
from ..fpbench import fpcast as ast
from .evalctx import EvalCtx
from . import ndarray

class EvaluatorError(utils.TitanicError):
    """Base Titanic evaluator error."""

class EvaluatorUnimplementedError(EvaluatorError):
    """Unimplemented feature in Titanic evaluator."""

class EvaluatorUnboundError(EvaluatorError, LookupError):
    """Unbound variable encountered during evaluation."""


class Evaluator(object):
    """FPCore evaluator.
    Dispatches on type of expressions in the AST.
    """

    ctype = EvalCtx

    @classmethod
    def _eval_expr(cls, e, ctx):
        raise EvaluatorUnimplementedError('expr {}: unimplemented'.format(str(e)))

    @classmethod
    def _eval_var(cls, e, ctx):
        raise EvaluatorUnimplementedError('var {}: unimplemented'.format(str(e)))

    @classmethod
    def _eval_val(cls, e, ctx):
        raise EvaluatorUnimplementedError('val {}: unimplemented'.format(str(e)))

    @classmethod
    def _eval_ctx(cls, e, ctx):
        # Note that let creates a new context, so the old one will
        # not be changed.
        return cls.evaluate(e.body, ctx.let(props=e.props))

    @classmethod
    def _eval_tensor(cls, e, ctx):
        raise EvaluatorUnimplementedError('tensor {}: unimplemented'.format(str(e)))

    @classmethod
    def _eval_if(cls, e, ctx):
        raise EvaluatorUnimplementedError('control {}: unimplemented'.format(str(e)))

    @classmethod
    def _eval_let(cls, e, ctx):
        raise EvaluatorUnimplementedError('control {}: unimplemented'.format(str(e)))

    @classmethod
    def _eval_while(cls, e, ctx):
        raise EvaluatorUnimplementedError('control {}: unimplemented'.format(str(e)))

    @classmethod
    def _eval_op(cls, e, ctx):
        raise EvaluatorUnimplementedError('op {}: unimplemented'.format(str(e)))

    @classmethod
    def _eval_unknown(cls, e, ctx):
        raise EvaluatorError('unknown operation {}'.format(e.name))

    _evaluator_dispatch = {
        # catch-all for otherwise unimplemented AST nodes
        ast.Expr: '_eval_expr',
        # unstructured data, which normally isn't evaluated
        ast.Data: '_eval_data',
        # variables are their own thing
        ast.Var: '_eval_var',
        # catch-all for all constant-valued expressions
        ast.Val: '_eval_val',
        # specific types of values
        ast.Constant: '_eval_constant',
        ast.Decnum: '_eval_decnum',
        ast.Hexnum: '_eval_hexnum',
        ast.Integer: '_eval_integer',
        ast.Rational: '_eval_rational',
        ast.Digits: '_eval_digits',
        # strings are special ValueExprs (not Vals) that won't normally be evaluated
        ast.String: '_eval_string',
        # rounding contexts
        ast.Ctx: '_eval_ctx',
        # Tensors
        ast.Tensor: '_eval_tensor',
        # control flow
        ast.If: '_eval_if',
        ast.Let: '_eval_let',
        ast.LetStar: '_eval_letstar',
        ast.While: '_eval_while',
        ast.WhileStar: '_eval_whilestar',
        # catch-all for operations with some number of arguments
        ast.NaryExpr: '_eval_op',
        # catch-all for operations not recognized by the compiler
        ast.UnknownOperator: '_eval_unknown',
        # specific operations
        ast.Cast: '_eval_cast',
        ast.Dim: '_eval_dim',
        ast.Size: '_eval_size',
        ast.Get: '_eval_get',
        ast.Add: '_eval_add',
        ast.Sub: '_eval_sub',
        ast.Mul: '_eval_mul',
        ast.Div: '_eval_div',
        ast.Sqrt: '_eval_sqrt',
        ast.Fma: '_eval_fma',
        ast.Neg: '_eval_neg',
        ast.Copysign: '_eval_copysign',
        ast.Fabs: '_eval_fabs',
        ast.Fdim: '_eval_fdim',
        ast.Fmax: '_eval_fmax',
        ast.Fmin: '_eval_fmin',
        ast.Fmod: '_eval_fmod',
        ast.Remainder: '_eval_remainder',
        ast.Ceil: '_eval_ceil',
        ast.Floor: '_eval_floor',
        ast.Nearbyint: '_eval_nearbyint',
        ast.Round: '_eval_round',
        ast.Trunc: '_eval_trunc',
        ast.Acos: '_eval_acos',
        ast.Acosh: '_eval_acosh',
        ast.Asin: '_eval_asin',
        ast.Asinh: '_eval_asinh',
        ast.Atan: '_eval_atan',
        ast.Atan2: '_eval_atan2',
        ast.Atanh: '_eval_atanh',
        ast.Cos: '_eval_cos',
        ast.Cosh: '_eval_cosh',
        ast.Sin: '_eval_sin',
        ast.Sinh: '_eval_sinh',
        ast.Tan: '_eval_tan',
        ast.Tanh: '_eval_tanh',
        ast.Exp: '_eval_exp',
        ast.Exp2: '_eval_exp2',
        ast.Expm1: '_eval_exmp1',
        ast.Log: '_eval_log',
        ast.Log10: '_eval_log10',
        ast.Log1p: '_eval_log1p',
        ast.Log2: '_eval_log2',
        ast.Cbrt: '_eval_cbrt',
        ast.Hypot: '_eval_hypot',
        ast.Pow: '_eval_pow',
        ast.Erf: '_eval_erf',
        ast.Erfc: '_eval_erfc',
        ast.Lgamma: '_eval_lgamma',
        ast.Tgamma: '_eval_tgamma',
        ast.LT: '_eval_lt',
        ast.GT: '_eval_gt',
        ast.LEQ: '_eval_leq',
        ast.GEQ: '_eval_geq',
        ast.EQ: '_eval_eq',
        ast.NEQ: '_eval_neq',
        ast.Isfinite: '_eval_isfinite',
        ast.Isinf: '_eval_isinf',
        ast.Isnan: '_eval_isnan',
        ast.Isnormal: '_eval_isnormal',
        ast.Signbit: '_eval_signbit',
        ast.And: '_eval_and',
        ast.Or: '_eval_or',
        ast.Not: '_eval_not',
    }

    _evaluator_cache = utils.ImmutableDict()

    # this is the sort of things that should be tracked in an instance
    # rather than implemented directly in the class...
    # evals = 0

    @classmethod
    def evaluate(cls, e, ctx):
        try:
            method = cls._evaluator_cache[type(e)]
        except KeyError:
            # initialize the cache for this class if it hasn't been initialized already
            if isinstance(cls._evaluator_cache, utils.ImmutableDict):
                cls._evaluator_cache = {}
            # walk up the mro and assign the evaluator for the first subtype to this type
            method = None
            ecls = type(e)
            for superclass in ecls.__mro__:
                method_name = cls._evaluator_dispatch.get(superclass, None)
                if method_name is not None and hasattr(cls, method_name):
                    method = getattr(cls, method_name)
                    cls._evaluator_cache[ecls] = method
                    break
            if method is None:
                raise EvaluatorError('Evaluator: unable to dispatch for expression {} with mro {}'
                                     .format(repr(e), repr(ecls.__mro__)))

        # cls.evals += 1
        # if cls.evals & 0xfffff == 0xfffff:
        #     print(',', end='', flush=True)


        # result = method(e, ctx)
        # print(repr(method))
        # print(str(e))
        # print(str(ctx.bindings))
        # print(' ->', str(result))
        # print()

        # return result

        return method(e, ctx)


class BaseInterpreter(Evaluator):
    """Base interpreter - only implements control flow."""

    # datatype conversion

    dtype = type(None)
    ctype = EvalCtx

    constants = {
        'TRUE': True,
        'FALSE': False,
    }

    @staticmethod
    def arg_to_digital(x, ctx):
        raise EvaluatorUnimplementedError('arg_to_digital({}): unimplemented'.format(repr(x)))

    @staticmethod
    def round_to_context(x, ctx):
        raise EvaluatorUnimplementedError('round_to_context({}): unimplemented'.format(repr(x)))


    # values

    @classmethod
    def _eval_var(cls, e, ctx):
        try:
            return ctx.bindings[e.value]
        except KeyError as exn:
            raise EvaluatorUnboundError(exn.args[0])

    @classmethod
    def _eval_val(cls, e, ctx):
        return cls.arg_to_digital(e.value, ctx)

    @classmethod
    def _eval_constant(cls, e, ctx):
        try:
            return cls.constants[e.value]
        except KeyError as exn:
            raise EvaluatorUnimplementedError('unsupported constant {}'.format(repr(exn.args[0])))

    @classmethod
    def _eval_data(cls, e, ctx):
        data, shape = ndarray.flatten_shaped_list(e.as_list())
        rounded_data = [cls.evaluate(d, ctx) for d in data]        
        return ndarray.NDArray(shape, data)

    # Tensors
        
    @classmethod
    def _eval_dim(cls, e, ctx):
        nd = cls.evaluate(e.children[0], ctx)
        if not isinstance(nd, ndarray.NDArray):
            raise EvaluatorError('{} must be a tensor to get its dimension'.format(repr(nd)))
        return digital.Digital(m=len(nd.shape), exp=0)

    @classmethod
    def _eval_size(cls, e, ctx):
        nd = cls.evaluate(e.children[0], ctx)
        if not isinstance(nd, ndarray.NDArray):
            raise EvaluatorError('{} must be a tensor to get the size of a dimension'.format(repr(nd)))
        idx = cls.evaluate(e.children[1], ctx)
        if not idx.is_integer():
            raise EvaluatorError('computed shape index {} must be an integer'.format(repr(idx)))
        i = int(idx.m * (2**idx.exp))
        return digital.Digital(m=nd.shape[i], exp=0)


    @classmethod
    def _eval_get(cls, e, ctx):
        nd = cls.evaluate(e.children[0], ctx)
        if not isinstance(nd, ndarray.NDArray):
            raise EvaluatorError('{} must be a tensor to get an element'.format(repr(nd)))
        pos = []
        for child in e.children[1:]:
            idx = cls.evaluate(child, ctx)
            if not idx.is_integer():
                raise EvaluatorError('computed index {} must be an integer'.format(repr(idx)))
            pos.append(int(idx.m * (2**idx.exp)))
        return nd[pos]

    @classmethod
    def _eval_tensor(cls, e, ctx):
        shape = []
        names = []
        for name, expr in e.dim_bindings:
            size = cls.evaluate(expr, ctx)
            if not size.is_integer():
                raise EvaluatorError('dimension size {} must be an integer'.format(repr(size)))
            # can you spot the bug in this arbitrary precision -> integer conversion?
            shape.append(int(size.m * (2**size.exp)))
            names.append(name)
            
        data = [cls.evaluate(e.body, ctx.let(bindings=[(name, digital.Digital(m=i,exp=0))
                                                       for name, i in zip(names, ndarray.position(shape, idx))]))
                for idx in range(ndarray.shape_size(shape))]
        return ndarray.NDArray(shape=shape, data=data)

    # control flow

    @classmethod
    def _eval_if(cls, e, ctx):
        if cls.evaluate(e.cond, ctx):
            return cls.evaluate(e.then_body, ctx)
        else:
            return cls.evaluate(e.else_body, ctx)

    @classmethod
    def _eval_let(cls, e, ctx):
        bindings = [(name, cls.evaluate(expr, ctx)) for name, expr in e.let_bindings]
        return cls.evaluate(e.body, ctx.let(bindings=bindings))

    # @classmethod
    # def _eval_letstar(cls, e, ctx):
    #     bindings = []
    #     for name, expr in e.let_bindings:
    #         local_ctx = ctx.let(bindings=bindings)
    #         bindings.append((name, cls.evaluate(expr, local_ctx)))
    #     return cls.evaluate(e.body, ctx.let(bindings=bindings))

    @classmethod
    def _eval_letstar(cls, e, ctx):
        for name, expr in e.let_bindings:
            new_binding = (name, cls.evaluate(expr, ctx))
            ctx = ctx.let(bindings=[new_binding])
        return cls.evaluate(e.body, ctx)

    @classmethod
    def _eval_while(cls, e, ctx):
        bindings = [(name, cls.evaluate(init_expr, ctx)) for name, init_expr, update_expr in e.while_bindings]
        ctx = ctx.let(bindings=bindings)
        while cls.evaluate(e.cond, ctx):
            bindings = [(name, cls.evaluate(update_expr, ctx)) for name, init_expr, update_expr in e.while_bindings]
            ctx = ctx.let(bindings=bindings)
        return cls.evaluate(e.body, ctx)

    @classmethod
    def _eval_whilestar(cls, e, ctx):
        for name, init_expr, update_expr in e.while_bindings:
            new_binding = (name, cls.evaluate(init_expr, ctx))
            ctx = ctx.let(bindings=[new_binding])
        while cls.evaluate(e.cond, ctx):
            for name, init_expr, update_expr in e.while_bindings:
                new_binding = (name, cls.evaluate(update_expr, ctx))
                ctx = ctx.let(bindings=[new_binding])
        return cls.evaluate(e.body, ctx)


    # interpreter interface

    @classmethod
    def arg_ctx(cls, core, args, ctx=None, override=True):
        if len(core.inputs) != len(args):
            raise ValueError('incorrect number of arguments: got {}, expecting {} ({})'.format(
                len(args), len(core.inputs), ' '.join((name for name, props in core.inputs))))

        if ctx is None:
            ctx = cls.ctype(props=core.props)
        elif override:
            allprops = {}
            allprops.update(core.props)
            allprops.update(ctx.props)
            ctx = ctx.let(props=allprops)
        else:
            ctx = ctx.let(props=core.props)

        arg_bindings = []

        for arg, (name, props, shape) in zip(args, core.inputs):
            local_ctx = ctx.let(props=props)

            if isinstance(arg, cls.dtype):
                argval = cls.round_to_context(arg, ctx=local_ctx)
            elif isinstance(arg, ast.Expr):
                argval = cls.evaluate(arg, local_ctx)
            else:
                argval = cls.arg_to_digital(arg, local_ctx)

            arg_bindings.append((name, argval))

        return ctx.let(bindings=arg_bindings)

    @classmethod
    def interpret(cls, core, args, ctx=None, override=True):
        ctx = cls.arg_ctx(core, args, ctx=ctx, override=override)
        return cls.evaluate(core.e, ctx)

    @classmethod
    def interpret_pre(cls, core, args, ctx=None, override=True):
        if core.pre is None:
            return True
        else:
            ctx = cls.arg_ctx(core, args, ctx=ctx, override=override)
            return cls.evaluate(core.pre, ctx)


class SimpleInterpreter(BaseInterpreter):
    """Simple FPCore interpreter.
    Override Interpreter.dtype with any class that supports
    simple arithmetic operators: a + b, a > b, abs(a).
    """

    @classmethod
    def _eval_cast(cls, e, ctx):
        return cls.round_to_context(cls.evaluate(e.children[0], ctx), ctx)

    @classmethod
    def _eval_add(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx) + cls.evaluate(e.children[1], ctx)

    @classmethod
    def _eval_sub(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx) - cls.evaluate(e.children[1], ctx)

    @classmethod
    def _eval_mul(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx) * cls.evaluate(e.children[1], ctx)

    # Division may need to be overwritten for types that raise an exception on
    # division by 0 (such as Python floats).

    @classmethod
    def _eval_div(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx) / cls.evaluate(e.children[1], ctx)

    @classmethod
    def _eval_neg(cls, e, ctx):
        return -cls.evaluate(e.children[0], ctx)

    @classmethod
    def _eval_fabs(cls, e, ctx):
        return abs(cls.evaluate(e.children[0], ctx))

    # In the common case where nan isn't ordered, it will probably be necessary
    # to override these functions so that they do the right thing.

    @classmethod
    def _eval_fmax(cls, e, ctx):
        return max(cls.evaluate(e.children[0], ctx), cls.evaluate(e.children[1], ctx))

    @classmethod
    def _eval_fmin(cls, e, ctx):
        return min(cls.evaluate(e.children[0], ctx), cls.evaluate(e.children[1], ctx))

    # This ugly code is designed to cause comparison operators to short-circuit.
    # A more elegant implementation:
    #    children = [cls.evaluate(child, ctx) for child in e.children]
    #    for x, y in zip(children, children[1:]):
    #        if not x < y:
    #            return False
    #    return True
    # does not short-circuit, which is inconsistent with the behavior of
    # _eval_and and _eval_or, which use the short-circuiting all and any builtins.

    @classmethod
    def _eval_lt(cls, e, ctx):
        if len(e.children) == 2:
            return cls.evaluate(e.children[0], ctx) < cls.evaluate(e.children[1], ctx)
        elif len(e.children) < 2:
            return True
        else:
            a = cls.evaluate(e.children[0], ctx)
            for child in e.children[1:]:
                b = cls.evaluate(child, ctx)
                if not a < b:
                    return False
                a = b
            return True

    @classmethod
    def _eval_gt(cls, e, ctx):
        if len(e.children) == 2:
            return cls.evaluate(e.children[0], ctx) > cls.evaluate(e.children[1], ctx)
        elif len(e.children) < 2:
            return True
        else:
            a = cls.evaluate(e.children[0], ctx)
            for child in e.children[1:]:
                b = cls.evaluate(child, ctx)
                if not a > b:
                    return False
                a = b
            return True

    @classmethod
    def _eval_leq(cls, e, ctx):
        if len(e.children) == 2:
            return cls.evaluate(e.children[0], ctx) <= cls.evaluate(e.children[1], ctx)
        elif len(e.children) < 2:
            return True
        else:
            a = cls.evaluate(e.children[0], ctx)
            for child in e.children[1:]:
                b = cls.evaluate(child, ctx)
                if not a <= b:
                    return False
                a = b
            return True

    @classmethod
    def _eval_geq(cls, e, ctx):
        if len(e.children) == 2:
            return cls.evaluate(e.children[0], ctx) >= cls.evaluate(e.children[1], ctx)
        elif len(e.children) < 2:
            return True
        else:
            a = cls.evaluate(e.children[0], ctx)
            for child in e.children[1:]:
                b = cls.evaluate(child, ctx)
                if not a >= b:
                    return False
                a = b
            return True

    @classmethod
    def _eval_eq(cls, e, ctx):
        if len(e.children) == 2:
            return cls.evaluate(e.children[0], ctx) == cls.evaluate(e.children[1], ctx)
        elif len(e.children) < 2:
            return True
        else:
            a = cls.evaluate(e.children[0], ctx)
            for child in e.children[1:]:
                b = cls.evaluate(child, ctx)
                if not a == b:
                    return False
                a = b
            return True

    # This is particularly complicated, because we need to try every combination
    # to ensure that each element is distinct. To short-circuit but avoid recomputation,
    # we keep a cache of results we've already computed. Note that we can't just
    # index with the child expressions themselves, or we might notice that shared
    # subtrees are actually the same object and forget to re-evaluate some of them.

    @classmethod
    def _eval_neq(cls, e, ctx):
        if len(e.children) == 2:
            return cls.evaluate(e.children[0], ctx) != cls.evaluate(e.children[1], ctx)
        elif len(e.children) < 2:
            return True
        else:
            nchildren = len(e.children)
            cached = [False] * nchildren
            cache = [None] * nchildren
            for i1, i2 in itertools.combinations(range(nchildren), 2):
                if cached[i1]:
                    a = cache[i1]
                else:
                    a = cls.evaluate(e.children[i1], ctx)
                    cache[i1] = a
                    cached[i1] = True
                if cached[i2]:
                    b = cache[i2]
                else:
                    b = cls.evaluate(e.children[i2], ctx)
                    cache[i2] = b
                    cached[i2] = True
                if not a != b:
                    return False
            return True

    @classmethod
    def _eval_and(cls, e, ctx):
        return all(cls.evaluate(child, ctx) for child in e.children)

    @classmethod
    def _eval_or(cls, e, ctx):
        return any(cls.evaluate(child, ctx) for child in e.children)

    @classmethod
    def _eval_not(cls, e, ctx):
        return not cls.evaluate(e.children[0], ctx)


class StandardInterpreter(SimpleInterpreter):
    """Standard FPCore interpreter.
    Override Interpreter.dtype with a class derived from Digital
    all standard FPCore operations as methods: x.add(y), a.fma(y, z), etc.
    i.e. MPNum.
    """

    @classmethod
    def _eval_add(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).add(cls.evaluate(e.children[1], ctx), ctx=ctx)

    @classmethod
    def _eval_sub(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).sub(cls.evaluate(e.children[1], ctx), ctx=ctx)

    @classmethod
    def _eval_mul(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).mul(cls.evaluate(e.children[1], ctx), ctx=ctx)

    @classmethod
    def _eval_div(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).div(cls.evaluate(e.children[1], ctx), ctx=ctx)

    @classmethod
    def _eval_sqrt(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).sqrt(ctx=ctx)

    @classmethod
    def _eval_fma(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).fma(cls.evaluate(e.children[1], ctx), cls.evaluate(e.children[2], ctx), ctx=ctx)

    @classmethod
    def _eval_neg(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).neg(ctx=ctx)

    @classmethod
    def _eval_copysign(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).copysign(cls.evaluate(e.children[1], ctx), ctx=ctx)

    @classmethod
    def _eval_fabs(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).fabs(ctx=ctx)

    @classmethod
    def _eval_fdim(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).fdim(cls.evaluate(e.children[1], ctx), ctx=ctx)

    @classmethod
    def _eval_fmax(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).fmax(cls.evaluate(e.children[1], ctx), ctx=ctx)

    @classmethod
    def _eval_fmin(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).fmin(cls.evaluate(e.children[1], ctx), ctx=ctx)

    @classmethod
    def _eval_fmod(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).fmod(cls.evaluate(e.children[1], ctx), ctx=ctx)

    @classmethod
    def _eval_remainder(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).remainder(cls.evaluate(e.children[1], ctx), ctx=ctx)

    @classmethod
    def _eval_ceil(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).ceil(ctx=ctx)

    @classmethod
    def _eval_floor(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).floor(ctx=ctx)

    @classmethod
    def _eval_nearbyint(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).nearbyint(ctx=ctx)

    @classmethod
    def _eval_round(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).round(ctx=ctx)

    @classmethod
    def _eval_trunc(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).trunc(ctx=ctx)

    @classmethod
    def _eval_acos(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).acos(ctx=ctx)

    @classmethod
    def _eval_acosh(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).acosh(ctx=ctx)

    @classmethod
    def _eval_asin(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).asin(ctx=ctx)

    @classmethod
    def _eval_asinh(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).asinh(ctx=ctx)

    @classmethod
    def _eval_atan(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).atan(ctx=ctx)

    @classmethod
    def _eval_atan2(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).atan2(cls.evaluate(e.children[1], ctx), ctx=ctx)

    @classmethod
    def _eval_atanh(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).atanh(ctx=ctx)

    @classmethod
    def _eval_cos(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).cos(ctx=ctx)

    @classmethod
    def _eval_cosh(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).cosh(ctx=ctx)

    @classmethod
    def _eval_sin(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).sin(ctx=ctx)

    @classmethod
    def _eval_sinh(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).sinh(ctx=ctx)

    @classmethod
    def _eval_tan(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).tan(ctx=ctx)

    @classmethod
    def _eval_tanh(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).tanh(ctx=ctx)

    @classmethod
    def _eval_exp(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).exp_(ctx=ctx)

    @classmethod
    def _eval_exp2(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).exp2(ctx=ctx)

    @classmethod
    def _eval_expm1(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).expm1(ctx=ctx)

    @classmethod
    def _eval_log(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).log(ctx=ctx)

    @classmethod
    def _eval_log10(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).log10(ctx=ctx)

    @classmethod
    def _eval_log1p(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).log1p(ctx=ctx)

    @classmethod
    def _eval_log2(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).log2(ctx=ctx)

    @classmethod
    def _eval_cbrt(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).cbrt(ctx=ctx)

    @classmethod
    def _eval_hypot(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).hypot(cls.evaluate(e.children[1], ctx), ctx=ctx)

    @classmethod
    def _eval_pow(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).pow(cls.evaluate(e.children[1], ctx), ctx=ctx)

    @classmethod
    def _eval_erf(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).erf(ctx=ctx)

    @classmethod
    def _eval_erfc(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).erfc(ctx=ctx)

    @classmethod
    def _eval_lgamma(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).lgamma(ctx=ctx)

    @classmethod
    def _eval_tgamma(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).tgamma(ctx=ctx)

    @classmethod
    def _eval_isfinite(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).isfinite()

    @classmethod
    def _eval_isinf(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).isinf

    @classmethod
    def _eval_isnan(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).isnan

    @classmethod
    def _eval_isnormal(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).isnormal()

    @classmethod
    def _eval_signbit(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).signbit()
