"""Direct softposit arithmetic. No emulation with MPFR.
"""

import math

from sfpy import Posit8, Posit16, Posit32

from . import interpreter
from . import evalctx
from ..titanic import gmpmath

softposit_precs = {
    (0, 8): Posit8,
    (1, 16): Posit16,
    (2, 32): Posit32,
}

class Interpreter(interpreter.SimpleInterpreter):
    """FPCore interpreter using sfpy wrapper for official SoftPosit library.
    Supports only the following posit formats:
    posit8 (es=0, nbits=8)
    posit16 (es=1, nbits=16)
    posit32 (es=3, nbits=32)
    Only basic operations implemented in the softposit library are supported
    (+ - * / fma sqrt).
    For now, there is no support for mixed precision or for the quire.
    """

    # datatype conversion

    dtype = type(None)
    ctype = evalctx.PositCtx

    # constants = {}
    # not supported

    @staticmethod
    def arg_to_digital(x, ctx):
        try:
            positcls = softposit_precs[(ctx.es, ctx.nbits)]
        except KeyError as exn:
            raise interpreter.EvaluatorError('unsupported posit format: es={:d}, nbits={:d}'
                             .format(ctx.es, ctx.nbits))
        # this may double round, sort of by design of the softposit library...
        return positcls(float(x))

    @staticmethod
    def round_to_context(x, ctx):
        try:
            positcls = softposit_precs[(ctx.es, ctx.nbits)]
        except KeyError as exn:
            raise interpreter.EvaluatorError('unsupported posit format: es={:d}, nbits={:d}'
                             .format(ctx.es, ctx.nbits))

        inputcls = type(x)
        if inputcls == positcls or isinstance(x, bool):
            return x
        elif positcls == Posit8:
            return x.to_p8()
        elif positcls == Posit16:
            return x.to_p16()
        else:
            return x.to_p32()


    # values

    @classmethod
    def _eval_decnum(cls, e, ctx):
        return cls.arg_to_digital(e.value, ctx)

    @classmethod
    def _eval_hexnum(cls, e, ctx):
        return cls.arg_to_digital(float.fromhex(e.value), ctx)

    @classmethod
    def _eval_integer(cls, e, ctx):
        return cls.arg_to_digital(e.i, ctx)

    @classmethod
    def _eval_rational(cls, e, ctx):
        # may double round
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

    # hacked together

    @classmethod
    def _eval_isfinite(cls, e, ctx):
        child0 = cls.evaluate(e.children[0], ctx)
        return child0 != type(child0)(float('inf'))

    @classmethod
    def _eval_isinf(cls, e, ctx):
        child0 = cls.evaluate(e.children[0], ctx)
        return child0 == type(child0)(float('inf'))

    @classmethod
    def _eval_isnan(cls, e, ctx):
        child0 = cls.evaluate(e.children[0], ctx)
        return child0 == type(child0)(float('inf'))

    @classmethod
    def _eval_isnormal(cls, e, ctx):
        return True

    @classmethod
    def _eval_signbit(cls, e, ctx):
        child0 = cls.evaluate(e.children[0], ctx)
        return child0 < type(child0)(0)
