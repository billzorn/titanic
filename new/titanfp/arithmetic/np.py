"""Use numpy to interpret FPCores.
"""

import numpy as np

from . import interpreter
from . import evalctx


np.seterr(all='ignore')

np_precs = {}
np_precs.update((k, np.float16) for k in evalctx.binary16_synonyms)
np_precs.update((k, np.float32) for k in evalctx.binary32_synonyms)
np_precs.update((k, np.float64) for k in evalctx.binary64_synonyms)


_SMALLEST_NORMALS = {
    np.float16: np.float16(2.0 ** -14),
    np.float32: np.float32(2.0 ** -126),
    np.float64: np.float64(2.0 ** -1022),
}


class Interpreter(interpreter.SimpleInterpreter):
    """FPCore interpreter using numpy.
    Supports 16, 32 and 64-bit floats. Mixed-precision operations
    automatically upcast to the largest used type, double-rounding
    for operations at lower precision.
    """

    # datatype conversion

    # note that this type cannot be created directly
    dtype = np.floating
    ctype = evalctx.EvalCtx

    constants = {
        'E': np.e,
        'LOG2E': np.log2(np.e),
        'LOG10E': np.log10(np.e),
        'LN2': np.log(2),
        'LN10': np.log(10),
        'PI': np.pi,
        'PI_2': np.pi / 2,
        'PI_4': np.pi / 4,
        '1_PI': 1 / np.pi,
        '2_PI': 2 / np.pi,
        '2_SQRTPI': 2 / np.sqrt(np.pi),
        'SQRT2': np.sqrt(2),
        'SQRT1_2': np.sqrt(1/2),
        'INFINITY': np.inf,
        'NAN': np.nan,
        'TRUE': True,
        'FALSE': False,
    }

    @staticmethod
    def arg_to_digital(x, ctx):
        prec = ctx.get('precision', 'binary64')
        try:
            return np_precs[prec](x)
        except KeyError as exn:
            raise ValueError('unsupported precision {}'.format(repr(exn.args[0])))

    @staticmethod
    def round_to_context(x, ctx):
        prec = ctx.get('precision', 'binary64')
        try:
            dtype = np_precs[prec]
            if type(x) is dtype:
                return x
            else:
                return dtype(x)
        except KeyError as exn:
            raise ValueError('unsupported precision {}'.format(repr(exn.args[0])))


    # values

    @classmethod
    def _eval_decnum(cls, e, ctx):
        # numpy's implementation of parsing strings as floats is believed to double round...
        return cls.arg_to_digital(e.value, ctx)

    @classmethod
    def _eval_hexnum(cls, e, ctx):
        # may double round
        return cls.round_to_context(float.fromhex(e.value), ctx)

    @classmethod
    def _eval_rational(cls, e, ctx):
        # may double round
        try:
            f = e.p / e.q
        except OverflowError:
            f = np.inf * np.copysign(1.0, e.p)
        return cls.round_to_context(f, ctx)

    @classmethod
    def _eval_digits(cls, e, ctx):
        # may double round twice for inexact values
        digits = compute_digits(e.m, e.e, e.b, prec=53)
        # TODO: not guaranteed correct rounding, return code is ignored!
        f = float(gmpmath.digital_to_mpfr(digits))
        # and... round again
        return cls.round_to_context(f, ctx)


    # arithmetic

    @classmethod
    def _eval_sqrt(cls, e, ctx):
        return np.sqrt(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_fma(cls, e, ctx):
        raise ValueError('this is ridiculous')

    @classmethod
    def _eval_copysign(cls, e, ctx):
        return np.copysign(cls.evaluate(e.children[0], ctx), cls.evaluate(e.children[1], ctx))

    @classmethod
    def _eval_fdim(cls, e, ctx):
        child0 = cls.evaluate(e.children[0], ctx)
        child1 = cls.evaluate(e.children[1], ctx)
        if child0 > child1:
            return child0 - child1
        else:
            return +0.0

    @classmethod
    def _eval_fmax(cls, e, ctx):
        return np.fmax(cls.evaluate(e.children[0], ctx), cls.evaluate(e.children[1], ctx))

    @classmethod
    def _eval_fmin(cls, e, ctx):
        return np.fmin(cls.evaluate(e.children[0], ctx), cls.evaluate(e.children[1], ctx))

    @classmethod
    def _eval_fmod(cls, e, ctx):
        return np.fmod(cls.evaluate(e.children[0], ctx), cls.evaluate(e.children[1], ctx))

    @classmethod
    def _eval_remainder(cls, e, ctx):
        return np.remainder(cls.evaluate(e.children[0], ctx), cls.evaluate(e.children[1], ctx))

    @classmethod
    def _eval_ceil(cls, e, ctx):
        return np.ceil(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_floor(cls, e, ctx):
        return np.floor(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_nearbyint(cls, e, ctx):
        return np.rint(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_round(cls, e, ctx):
        raise ValueError('also ridiculous')

    @classmethod
    def _eval_trunc(cls, e, ctx):
        return np.trunc(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_acos(cls, e, ctx):
        return np.arccos(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_acosh(cls, e, ctx):
        return np.arccosh(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_asin(cls, e, ctx):
        return np.arcsin(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_asinh(cls, e, ctx):
        return np.arcsinh(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_atan(cls, e, ctx):
        return np.arctan(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_atan2(cls, e, ctx):
        return np.arctan2(cls.evaluate(e.children[0], ctx), cls.evaluate(e.children[1], ctx))

    @classmethod
    def _eval_atanh(cls, e, ctx):
        return np.arctanh(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_cos(cls, e, ctx):
        return np.cos(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_cosh(cls, e, ctx):
        return np.cosh(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_sin(cls, e, ctx):
        return np.sinh(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_sinh(cls, e, ctx):
        return np.sinh(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_tan(cls, e, ctx):
        return np.tan(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_tanh(cls, e, ctx):
        return np.tanh(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_exp(cls, e, ctx):
        return np.exp(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_exp2(cls, e, ctx):
        return np.exp2(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_expm1(cls, e, ctx):
        return np.expm1(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_log(cls, e, ctx):
        return np.log(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_log10(cls, e, ctx):
        return np.log10(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_log1p(cls, e, ctx):
        return np.log1p(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_log2(cls, e, ctx):
        return np.log2(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_cbrt(cls, e, ctx):
        return np.cbrt(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_hypot(cls, e, ctx):
        return np.hypot(cls.evaluate(e.children[0], ctx), cls.evaluate(e.children[1], ctx))

    @classmethod
    def _eval_pow(cls, e, ctx):
        return np.power(cls.evaluate(e.children[0], ctx), cls.evaluate(e.children[1], ctx))

    @classmethod
    def _eval_hypot(cls, e, ctx):
        return np.hypot(cls.evaluate(e.children[0], ctx), cls.evaluate(e.children[1], ctx))

    @classmethod
    def _eval_erf(cls, e, ctx):
        return np.erf(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_erfc(cls, e, ctx):
        raise ValueError('erfc: no numpy implementation, emulation unsupported')

    @classmethod
    def _eval_lgamma(cls, e, ctx):
        raise ValueError('lgamma: no numpy implementation, emulation unsupported')

    @classmethod
    def _eval_tgamma(cls, e, ctx):
        raise ValueError('tgamma: no numpy implementation, emulation unsupported')

    @classmethod
    def _eval_isfinite(cls, e, ctx):
        return np.isfinite(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_isinf(cls, e, ctx):
        return np.isinf(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_isnan(cls, e, ctx):
        return np.isnan(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_isnormal(cls, e, ctx):
        child0 = cls.evaluate(e.children[0], ctx)
        prec = ctx.get('precision', 'binary64')
        try:
            dtype = np_precs[prec]
            smallest = _SMALLEST_NORMALS[dtype]
        except KeyError as exn:
            raise ValueError('unsupported precision or type {}'.format(repr(exn.args[0])))
        return ((not np.isnan(child0)) and
                (not np.isinf(child0)) and
                (not abs(child0) < smallest))

    @classmethod
    def _eval_signbit(cls, e, ctx):
        return np.signbit(cls.evaluate(e.children[0], ctx))

    @classmethod
    def evaluate(cls, e, ctx):
        result = super().evaluate(e, ctx)
        return cls.round_to_context(result, ctx)
