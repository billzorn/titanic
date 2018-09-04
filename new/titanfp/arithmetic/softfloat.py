"""Direct softfloat arithmetic. No emulation with MPFR.
"""

import math

from sfpy import Float16, Float32, Float64

from . import interpreter
from . import evalctx
from ..titanic import gmpmath

softfloat_precs = {
    (5, 11): Float16,
    (8, 24): Float32,
    (11, 53): Float64,
}

_SMALLEST_NORMALS = {
    Float16: Float16(2.0 ** -14),
    Float32: Float32(2.0 ** -126),
    Float64: Float64(2.0 ** -1022),
}

_BIT_SIZES = {
    Float16: 16,
    Float32: 32,
    Float64: 64,
}

class Interpreter(interpreter.SimpleInterpreter):
    """FPCore interpreter using sfpy wrapper for official softfloat library.
    Supports only the following IEEE 754 floating-point formats:
    binary16 (w=5, p=11)
    binary32 (w=8, p=24)
    binary64 (w=11, p=53)
    Only basic operations implemented in the softfloat library are supported
    (+ - * / fma sqrt).
    For now, there is no support for mixed precision computation.
    """

    # datatype conversion
    dtype = type(None)
    ctype = evalctx.IEEECtx

    # constants = {}
    # not supported

    @staticmethod
    def arg_to_digital(x, ctx):
        try:
            floatcls = softfloat_precs[(ctx.w, ctx.p)]
        except KeyError as exn:
            raise ValueError('unsupported float format: w={:d}, p={:d}'
                             .format(ctx.es, ctx.nbits))
        # this may double round for types other than float64 (just like numpy...)
        return floatcls(float(x))

    @staticmethod
    def round_to_context(x, ctx):
        try:
            floatcls = softfloat_precs[(ctx.w, ctx.p)]
        except KeyError as exn:
            raise ValueError('unsupported float format: w={:d}, p={:d}'
                             .format(ctx.w, ctx.p))

        inputcls = type(x)
        if inputcls == floatcls or isinstance(x, bool):
            return x
        elif floatcls == Float16:
            return x.to_f16()
        elif floatcls == Float32:
            return x.to_f32()
        else:
            return x.to_f64()


    # values

    @classmethod
    def _eval_decnum(cls, e, ctx):
        return cls.arg_to_digital(e.value, ctx)

    @classmethod
    def _eval_hexnum(cls, e, ctx):
        return cls.arg_to_digital(float.fromhex(e.value), ctx)

    @classmethod
    def _eval_rational(cls, e, ctx):
        # may double round for types other than float64
        try:
            f = e.p / e.q
        except OverflowError:
            f = math.inf * math.copysign(1.0, e.p)
        return cls.arg_to_digital(f, ctx)

    @classmethod
    def _eval_digits(cls, e, ctx):
        # may double round twice for inexact values
        digits = gmpmath.compute_digits(e.m, e.e, e.b, prec=53)
        # TODO: not guaranteed correct rounding, return code is ignored!
        f = float(gmpmath.digital_to_mpfr(digits))
        # and... round again
        return cls.arg_to_digital(f, ctx)


    # arithmetic

    @classmethod
    def _eval_sqrt(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx).sqrt()

    @classmethod
    def _eval_fma(cls, e, ctx):
        # the order is different for fpcore and sfpy:
        # fpcore: fma(a, b, c) = a * b + c
        # sfpy: fma(a, b, c) = a + b * c
        return cls.evaluate(e.children[2], ctx).fma(cls.evaluate(e.children[0], ctx), cls.evaluate(e.children[1], ctx))

    @classmethod
    def _eval_nearbyint(cls, e, ctx):
        return round(cls.evaluate(e.children[0], ctx))

    # sort of hacked together


    @classmethod
    def _eval_isfinite(cls, e, ctx):
        child0 = cls.evaluate(e.children[0], ctx)
        return child0 == child0 and abs(child0) != type(child0)(float('inf'))

    @classmethod
    def _eval_isinf(cls, e, ctx):
        child0 = cls.evaluate(e.children[0], ctx)
        return abs(child0) == type(child0)(float('inf'))

    @classmethod
    def _eval_isnan(cls, e, ctx):
        child0 = cls.evaluate(e.children[0], ctx)
        return child0 != child0

    @classmethod
    def _eval_isnormal(cls, e, ctx):
        child0 = cls.evaluate(e.children[0], ctx)
        floatcls = type(child0)
        try:
            smallest = _SMALLEST_NORMALS[floatcls]
        except KeyError as exn:
            raise ValueError('unsupported precision or type {}'.format(repr(exn.args[0])))
        return ((not np.isnan(child0)) and
                (not np.isinf(child0)) and
                (not abs(child0) < smallest))

    @classmethod
    def _eval_signbit(cls, e, ctx):
        child0 = cls.evaluate(e.children[0], ctx)
        floatcls = type(child0)
        try:
            bitsize = _BIT_SIZES[floatcls]
        except KeyError as exn:
            raise ValueError('unsupported precision or type {}'.format(repr(exn.args[0])))
        return child0.bits >> (bitsize - 1) != 0
