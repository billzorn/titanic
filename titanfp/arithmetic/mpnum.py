from abc import abstractmethod
from ..titanic import digital, gmpmath
from ..titanic.ops import OP

class MPNum(digital.Digital):

    # must be implemented in subclasses
    @abstractmethod
    def _select_context(cls, *args, ctx=None):
        raise ValueError('virtual method: unimplemented')

    @abstractmethod
    def _round_to_context(cls, unrounded, ctx=None, strict=False):
        raise ValueError('virtual method: unimplemented')

    # most operations

    def add(self, other, ctx=None):
        ctx = self._select_context(self, other, ctx=ctx)
        result = gmpmath.compute(OP.add, self, other, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def sub(self, other, ctx=None):
        ctx = self._select_context(self, other, ctx=ctx)
        result = gmpmath.compute(OP.sub, self, other, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def mul(self, other, ctx=None):
        ctx = self._select_context(self, other, ctx=ctx)
        result = gmpmath.compute(OP.mul, self, other, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def div(self, other, ctx=None):
        ctx = self._select_context(self, other, ctx=ctx)
        result = gmpmath.compute(OP.div, self, other, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def sqrt(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.sqrt, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def fma(self, other1, other2, ctx=None):
        ctx = self._select_context(self, other1, other2, ctx=ctx)
        result = gmpmath.compute(OP.fma, self, other1, other2, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def neg(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.neg, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def copysign(self, other, ctx=None):
        ctx = self._select_context(self, other, ctx=ctx)
        result = gmpmath.compute(OP.copysign, self, other, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def fabs(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.fabs, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def fdim(self, other, ctx=None):
        # emulated
        ctx = self._select_context(self, other, ctx=ctx)
        result = gmpmath.compute(OP.sub, self, other, prec=ctx.p)
        zero = digital.Digital(negative=False, c=0, exp=0)
        if result < zero:
            return type(self)(negative=False, c=0, exp=0, inexact=False, rc=0)
        else:
            # never return negative zero
            rounded = self._round_to_context(result, ctx=ctx, strict=True)
            return type(self)(rounded, negative=False)

    def fmax(self, other, ctx=None):
        # emulated
        ctx = self._select_context(self, other, ctx=ctx)
        if self.isnan:
            return self._round_to_context(other, ctx=ctx, strict=False)
        elif other.isnan:
            return self._round_to_context(self, ctx=ctx, strict=False)
        else:
            return self._round_to_context(max(self, other), ctx=ctx, strict=False)

    def fmin(self, other, ctx=None):
        # emulated
        ctx = self._select_context(self, other, ctx=ctx)
        if self.isnan:
            return self._round_to_context(other, ctx=ctx, strict=False)
        elif other.isnan:
            return self._round_to_context(self, ctx=ctx, strict=False)
        else:
            return self._round_to_context(min(self, other), ctx=ctx, strict=False)

    def fmod(self, other, ctx=None):
        ctx = self._select_context(self, other, ctx=ctx)
        result = gmpmath.compute(OP.fmod, self, other, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def remainder(self, other, ctx=None):
        ctx = self._select_context(self, other, ctx=ctx)
        result = gmpmath.compute(OP.remainder, self, other, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def ceil(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.ceil, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def floor(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.floor, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def nearbyint(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.nearbyint, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def round(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.round, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def trunc(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.trunc, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def acos(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.acos, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def acosh(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.acosh, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def asin(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.asin, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def asinh(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.asinh, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def atan(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.atan, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def atan2(self, other, ctx=None):
        ctx = self._select_context(self, other, ctx=ctx)
        result = gmpmath.compute(OP.atan2, self, other, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def atanh(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.atanh, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def cos(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.cos, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def cosh(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.cosh, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def sin(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.sin, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def sinh(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.sinh, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def tan(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.tan, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def tanh(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.tanh, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def exp_(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.exp, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def exp2(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.exp2, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def expm1(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.expm1, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def log(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.log, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def log10(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.log10, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def log1p(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.log1p, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def log2(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.log2, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def cbrt(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.cbrt, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def hypot(self, other, ctx=None):
        ctx = self._select_context(self, other, ctx=ctx)
        result = gmpmath.compute(OP.hypot, self, other, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def pow(self, other, ctx=None):
        ctx = self._select_context(self, other, ctx=ctx)
        if other.is_zero():
            # avoid possibly passing nan to gmpmath.compute
            return type(self)(negative=False, c=1, exp=0, inexact=False, rc=0)
        result = gmpmath.compute(OP.pow, self, other, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def erf(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.erf, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def erfc(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.erfc, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def lgamma(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.lgamma, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def tgamma(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.tgamma, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def isfinite(self):
        return not (self.isinf or self.isnan)

    # isinf and isnan are properties

    # isnormal is implementation specific - override if necessary
    def isnormal(self):
        return not (
            self.is_zero()
            or self.isinf
            or self.isnan
        )

    def signbit(self):
        return self.negative
