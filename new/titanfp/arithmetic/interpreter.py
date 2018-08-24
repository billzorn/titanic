"""Base FPCore interpreter, using Python floats as its datatype."""

import math

from ..fpbench import fpcast as ast
from ..titanic import gmpmath
from .evalctx import EvalCtx

class Interpreter(object):

    # datatype conversion

    dtype = float
    ctype = EvalCtx

    @classmethod
    def arg_to_digital(cls, x, ctx):
        return float(x)


    # values and control flow

    @classmethod
    def _eval_val(cls, e, ctx):
        s = e.value.upper()
        if s == 'TRUE':
            return True
        elif s == 'FALSE':
            return False
        else:
            return cls.arg_to_digital(e.value, ctx)

    @classmethod
    def _eval_var(cls, e, ctx):
        return ctx.bindings[e.value]

    @classmethod
    def _eval_digits(cls, e, ctx):
        m, e, b = e.m, int(e.e), int(e.b)
        digits = compute_digits(m, e, b, prec=53)
        # TODO: not guaranteed correct rounding, return code is ignored!
        return float(gmpmath.digital_to_mpfr(digits))

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


    # arithmetic

    @classmethod
    def _eval_cast(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx)

    @classmethod
    def _eval_add(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx) + cls.evaluate(e.children[1], ctx)


    # main evaluator / dispatch

    _evaluator_dispatch = {
        ast.Ctx: '_eval_ctx',
        ast.Val: '_eval_val',
        ast.Var: '_eval_var',
        ast.Digits: '_eval_digits',
        ast.If: '_eval_if',
        ast.Let: '_eval_let',
        ast.While: '_eval_while',
        ast.Cast: '_eval_cast',
        ast.Add: '_eval_add',
        ast.Sub: '_eval_sub',
        ast.Mul: '_eval_mul',
        ast.Div: '_eval_div',
        ast.Sqrt: '_eval_div',
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

    @classmethod
    def interpret(cls, core, args, ctx=None):
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
            else:
                argval = cls.arg_to_digital(arg, local_ctx)

            arg_bindings.append((name, argval))

        return cls.evaluate(core.e, ctx.let(bindings=arg_bindings))
