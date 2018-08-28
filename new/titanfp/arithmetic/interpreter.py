"""Base FPCore interpreter."""


import itertools

from ..fpbench import fpcast as ast
from .evalctx import EvalCtx


class BaseInterpreter(object):
    """Base interpreter - only implements control flow."""

    # datatype conversion

    dtype = None
    ctype = EvalCtx

    constants = {
        'TRUE': True,
        'FALSE': False,
    }

    @staticmethod
    def arg_to_digital(x, ctx):
        raise ValueError('BaseInterpreter: arg_to_digital({}): unimplemented'
                         .format(repr(x)))

    @staticmethod
    def round_to_context(x, ctx):
        raise ValueError('BaseInterpreter: round_to_context({}): unimplemented'
                         .format(repr(x)))


    # values

    @classmethod
    def _eval_var(cls, e, ctx):
        try:
            return ctx.bindings[e.value]
        except KeyError as exn:
            raise ValueError('unbound variable {}'.format(repr(exn.args[0])))

    # not called directly in interpreter,
    @classmethod
    def _eval_val(cls, e, ctx):
        return cls.arg_to_digital(e.value)

    @classmethod
    def _eval_constant(cls, e, ctx):
        raise ValueError('BaseInterpreter: val {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_decnum(cls, e, ctx):
        raise ValueError('BaseInterpreter: val {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_hexnum(cls, e, ctx):
        raise ValueError('BaseInterpreter: val {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_rational(cls, e, ctx):
        raise ValueError('BaseInterpreter: val {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_digits(cls, e, ctx):
        raise ValueError('BaseInterpreter: val {}: unimplemented'.format(repr(e)))


    # control flow

    @classmethod
    def _eval_ctx(cls, e, ctx):
        # Note that let creates a new context, so the old one will
        # not be changed.
        return cls.evaluate(e.body, ctx.let(props=e.props))

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

    @classmethod
    def _eval_while(cls, e, ctx):
        bindings = [(name, cls.evaluate(init_expr, ctx)) for name, init_expr, update_expr in e.while_bindings]
        ctx = ctx.let(bindings=bindings)
        while evaluate(e.cond, ctx):
            bindings = [(name, cls.evaluate(update_expr, ctx)) for name, init_expr, update_expr in e.while_bindings]
            ctx = ctx.let(bindings=bindings)
        return cls.evaluate(e.body, ctx)


    # operations

    @classmethod
    def _eval_cast(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_add(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_sub(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_mul(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_div(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_sqrt(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_fma(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_neg(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_copysign(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_fabs(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_fdim(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_fmax(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_fmin(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_fmod(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_remainder(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_ceil(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_floor(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_nearbyint(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_round(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_trunc(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_acos(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_acosh(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_asin(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_asinh(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_atan(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_atan2(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_atanh(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_cos(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_cosh(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_sin(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_sinh(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_tan(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_tanh(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_exp(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_exp2(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_expm1(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_log(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_log10(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_log1p(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_log2(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_cbrt(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_hypot(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_pow(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_erf(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_erfc(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_lgamma(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_tgamma(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_lt(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_gt(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_leq(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_geq(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_eq(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_neq(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_isfinite(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_isinf(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_isnan(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_isnormal(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_signbit(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_and(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_or(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))

    @classmethod
    def _eval_not(cls, e, ctx):
        raise ValueError('BaseInterpreter: eval {}: unimplemented'.format(repr(e)))


    # main evaluator / dispatch

    _evaluator_dispatch = {
        ast.Var: '_eval_var',
        ast.Constant: '_eval_constant',
        ast.Decnum: '_eval_decnum',
        ast.Hexnum: '_eval_hexnum',
        ast.Rational: '_eval_rational',
        ast.Digits: '_eval_digits',
        ast.Ctx: '_eval_ctx',
        ast.If: '_eval_if',
        ast.Let: '_eval_let',
        ast.While: '_eval_while',
        ast.Cast: '_eval_cast',
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

    @classmethod
    def evaluate(cls, e, ctx):
        return getattr(cls, cls._evaluator_dispatch[type(e)])(e, ctx)


    # interpreter interface

    @classmethod
    def arg_ctx(cls, core, args, ctx=None):
        if len(core.inputs) != len(args):
            raise ValueError('incorrect number of arguments: got {}, expecting {} ({})'.format(
                len(args), len(core.inputs), ' '.join((name for name, props in core.inputs))))

        if ctx is None:
            ctx = cls.ctype(props=core.props)

        arg_bindings = []

        for arg, (name, props) in zip(args, core.inputs):
            local_ctx = ctx.let(props=props)

            if isinstance(arg, cls.dtype):
                argval = arg
            elif isinstance(arg, ast.Val):
                try:
                    argval = cls.evaluate(arg, ctx)
                except Exception:
                    argval = cls._eval_val(arg, ctx)
            else:
                argval = cls.arg_to_digital(arg, local_ctx)

            arg_bindings.append((name, argval))

        return ctx.let(bindings=arg_bindings)

    @classmethod
    def interpret(cls, core, args, ctx=None):
        ctx = cls.arg_ctx(core, args, ctx=ctx)
        return cls.evaluate(core.e, ctx)

    @classmethod
    def interpret_pre(cls, core, args, ctx=None):
        if core.pre is None:
            return True
        else:
            ctx = cls.arg_ctx(core, args, ctx=ctx)
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
            for child in cls.children[1:]:
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
            for child in cls.children[1:]:
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
            for child in cls.children[1:]:
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
            for child in cls.children[1:]:
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
            for child in cls.children[1:]:
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


class Interpreter(SimpleInterpreter):
    """Standard FPCore interpreter.
    Override Interpreter.dtype with a class that supports
    all arithmetic operations: a ** b, a.fma(b, c), a.sin(), a.isinf().
    """

    @classmethod
    def _eval_sqrt(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).sqrt()

    @classmethod
    def _eval_fma(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).fma(cls.evaluate(e.children[1], ctx), cls.evaluate(e.children[2], ctx))

    @classmethod
    def _eval_copysign(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).copysign(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_fdim(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).fdim(cls.evaluate(e.children[1], ctx))

    @classmethod
    def _eval_fmod(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).fmod(cls.evaluate(e.children[1], ctx))

    @classmethod
    def _eval_remainder(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).remainder(cls.evaluate(e.children[1], ctx))

    @classmethod
    def _eval_ceil(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).ceil()

    @classmethod
    def _eval_floor(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).floor()

    @classmethod
    def _eval_nearbyint(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).nearbyint()

    @classmethod
    def _eval_round(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).round()

    @classmethod
    def _eval_trunc(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).trunc()

    @classmethod
    def _eval_acos(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).acos()

    @classmethod
    def _eval_acosh(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).acosh()

    @classmethod
    def _eval_asin(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).asin()

    @classmethod
    def _eval_asinh(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).asinh()

    @classmethod
    def _eval_atan(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).atan()

    @classmethod
    def _eval_atan2(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).atan2(cls.evaluate(e.children[1], ctx))

    @classmethod
    def _eval_atanh(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).atanh()

    @classmethod
    def _eval_cos(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).cos()

    @classmethod
    def _eval_cosh(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).cosh()

    @classmethod
    def _eval_sin(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).sin()

    @classmethod
    def _eval_sinh(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).sinh()

    @classmethod
    def _eval_tan(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).tan()

    @classmethod
    def _eval_tanh(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).tanh()

    @classmethod
    def _eval_exp(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).exp()

    @classmethod
    def _eval_exp2(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).exp2()

    @classmethod
    def _eval_expm1(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).expm1()

    @classmethod
    def _eval_log(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).log()

    @classmethod
    def _eval_log10(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).log10()

    @classmethod
    def _eval_log1p(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).log1p()

    @classmethod
    def _eval_log2(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).log2()

    @classmethod
    def _eval_cbrt(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).cbrt()

    @classmethod
    def _eval_hypot(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).hypot(cls.evaluate(e.children[1], ctx))

    @classmethod
    def _eval_pow(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx) ** cls.evaluate(e.children[1], ctx)

    @classmethod
    def _eval_erf(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).erf()

    @classmethod
    def _eval_erfc(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).erfc()

    @classmethod
    def _eval_lgamma(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).lgamma()

    @classmethod
    def _eval_tgamma(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).tgamma()

    @classmethod
    def _eval_isfinite(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).isfinite()

    @classmethod
    def _eval_isinf(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).isinf()

    @classmethod
    def _eval_isnan(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).isnan()

    @classmethod
    def _eval_isnormal(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).isnormal()

    @classmethod
    def _eval_signbit(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).signbit()
