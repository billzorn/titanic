"""Emulated Posit arithmetic.
"""

from ..titanic import gmpmath
from ..titanic import digital

from ..titanic.integral import bitmask
from ..titanic.ops import RM, OP
from .evalctx import PositCtx
from . import interpreter
from . import posit


class SinkingPosit(digital.Digital):

    _ctx : PositCtx = posit.posit_ctx(4, 64)

    @property
    def ctx(self):
        """The rounding context used to compute this value.
        If a computation takes place between two values, then
        it will either use a provided context (which will be recorded
        on the result) or the more precise of the parent contexts
        if none is provided. For posits, a context is more "precise"
        if it has greater nbits, or greater es if nbits is the same.
        """
        return self._ctx

    def is_identical_to(self, other):
        if isinstance(other, type(self)):
            return super().is_identical_to(other) and self.ctx.es == other.ctx.es and self.ctx.nbits == other.ctx.nbits
        else:
            return super().is_identical_to(other)

    def __init__(self, x=None, ctx=None, **kwargs):
        if ctx is None:
            ctx = type(self)._ctx

        if x is None or isinstance(x, digital.Digital):
            super().__init__(x=x, **kwargs)
        else:
            f = gmpmath.mpfr(x, ctx.p)
            unrounded = gmpmath.mpfr_to_digital(f)
            super().__init__(x=self._round_to_context(unrounded, ctx=ctx, strict=True), **kwargs)

        self._ctx = posit.posit_ctx(ctx.es, ctx.nbits)

    def __repr__(self):
        return '{}(negative={}, c={}, exp={}, inexact={}, rc={}, isinf={}, isnan={}, ctx={})'.format(
            type(self).__name__, repr(self._negative), repr(self._c), repr(self._exp),
            repr(self._inexact), repr(self._rc), repr(self._isinf), repr(self._isnan), repr(self._ctx)
        )

    def __str__(self):
        if self.isnan or self.isinf:
            return 'nar'

        elif self.inexact:
            d1, d2 = gmpmath.digital_to_dec_range(self)
            fstr = gmpmath.dec_range_to_str(d1, d2, scientific=False)
            estr = gmpmath.dec_range_to_str(d1, d2, scientific=True)
        else:
            d = gmpmath.Dec(self)
            fstr = d.string
            estr = d.estring

        if len(fstr) <= 16:
            return fstr
        elif len(fstr) <= len(estr):
            return fstr
        else:
            return estr

    def __float__(self):
        return float(gmpmath.digital_to_mpfr(self))

    @classmethod
    def _select_context(cls, *args, ctx=None):
        if ctx is not None:
            return posit.posit_ctx(ctx.es, ctx.nbits)
        else:
            nbits = max((f.ctx.nbits for f in args if isinstance(f, cls)))
            es = max((f.ctx.es for f in args if isinstance(f, cls) and f.ctx.nbits == nbits))
            return posit.posit_ctx(es, nbits)

    @classmethod
    def _round_to_context(cls, unrounded, max_p=None, min_n=None, inexact=None, ctx=None, strict=False):
        if ctx is None:
            if isinstance(unrounded, cls):
                ctx = unrounded.ctx
            else:
                raise ValueError('no context specified to round {}'.format(repr(unrounded)))


        if unrounded.isinf or unrounded.isnan:
            # all non-real values go to the single posit infinite value
            return cls(isnan=True, ctx=ctx)

        else:
            regime, e = divmod(unrounded.e, ctx.u)
            if regime < 0:
                rbits = -regime + 1
            else:
                rbits = regime + 2

            sbits = ctx.nbits - 1 - rbits - ctx.es

            # regime = max(abs(unrounded.e) - 1, 0) // ctx.u
            # sbits = ctx.nbits - 3 - ctx.es - regime

            if sbits < -ctx.es:
                # we are outside the representable range: return max / min
                if unrounded.e < 0:
                    rounded = digital.Digital(negative=unrounded.negative, c=1, exp=ctx.emin, inexact=True, rc=-1)
                else:
                    rounded = digital.Digital(negative=unrounded.negative, c=1, exp=ctx.emax, inexact=True, rc=1)

            elif sbits < 0:
                # round -sbits bits off of the exponent, because they won't fit
                offset = -sbits

                lost_bits = unrounded.e & bitmask(offset)
                # note these left bits might be negative
                left_bits = unrounded.e >> offset

                if offset > 0:
                    offset_m1 = offset - 1
                    low_bits = lost_bits & bitmask(offset_m1)
                    half_bit = lost_bits >> offset_m1
                else:
                    low_bits = 0
                    half_bit = 0

                lost_sig_bits = unrounded.c & bitmask(unrounded.c.bit_length() - 1)

                new_exp = left_bits
                if lost_bits > 0 or lost_sig_bits > 0 or unrounded.rc > 0:
                    rc = 1
                    exp_inexact = True
                else:
                    rc = unrounded.rc
                    exp_inexact = unrounded.inexact

                #print('->\t', new_exp, rc, exp_inexact)

                # We want to round on the geometric mean of the two numbers,
                # but this is the same as rounding on the arithmetic mean of
                # the exponents.

                if half_bit > 0:
                    if low_bits > 0 or lost_sig_bits > 0 or unrounded.rc > 0:
                        # round the exponent up; remember it might be negative, but that's ok
                        #print('UP')
                        new_exp += 1
                        rc = -1
                    elif unrounded.rc < 0:
                        # tie broken the other way
                        #print('RC DOWN')
                        pass
                    else:
                        # "round nearest, ties to arbitrary"
                        #print('TIE')

                        # just generate the bits and see if that's even
                        tmp_exp = new_exp << offset
                        rounded = digital.Digital(negative=unrounded.negative, c=1, exp=tmp_exp, inexact=exp_inexact, rc=rc)
                        bits = posit.digital_to_bits(rounded, ctx)
                        if bits & 1:
                            new_exp += 1
                            rc = -1

                new_exp <<= offset
                rounded = digital.Digital(negative=unrounded.negative, c=1, exp=new_exp, inexact=exp_inexact, rc=rc)

            elif sbits == 0:
                #print('special round')
                # round "normally", but with the weird behavior for ties
                rounded = unrounded.round_m(max_p=sbits + 1, min_n=None, rm=RM.RNE, strict=strict)

                # this is always rounding to one bit
                if unrounded.c == 0:
                    left_bits = 0
                    half_bit = 0
                    low_bits = 0
                else:
                    cbits = unrounded.c.bit_length()
                    left_bits = 1
                    half_bit = (unrounded.c >> (cbits - 2)) & 1
                    low_bits = unrounded.c & bitmask(cbits - 2)

                new_exp = unrounded.e
                if half_bit > 0 or low_bits > 0 or unrounded.rc > 0:
                    rc = 1
                    exp_inexact = True
                else:
                    rc = unrounded.rc
                    exp_inexact = unrounded.inexact

                if half_bit > 0:
                    if low_bits > 0 or unrounded.rc > 0:
                        # UP
                        new_exp += 1
                        rc = -1
                    elif unrounded.rc < 0:
                        # DOWN
                        pass
                    else:
                        # TIE
                        tmp_exp = new_exp
                        rounded = digital.Digital(negative=unrounded.negative, c=1, exp=tmp_exp, inexact=exp_inexact, rc=rc)
                        bits = posit.digital_to_bits(rounded, ctx)
                        if bits & 1:
                            new_exp += 1
                            rc = -1

                rounded = digital.Digital(negative=unrounded.negative, c=1, exp=new_exp, inexact=exp_inexact, rc=rc)


            else:
                #print('normal round')
                # we can represent the entire exponent, so only round the mantissa
                if max_p is None:
                    effective_prec = sbits+1
                else:
                    effective_prec = max(1, min(sbits+1, max_p))
                rounded = unrounded.round_m(max_p=effective_prec, min_n=min_n, rm=RM.RNE, strict=strict)

        # Posits do not have a signed zero, and never round down to zero.

        if inexact is not None and inexact != rounded.inexact:
            rounded = type(rounded)(rounded, inexact=inexact)

        if rounded.is_zero():
            if rounded.inexact:
                return cls(negative=rounded.negative, c=1, exp=rounded.exp, rc=-1, ctx=ctx)
            else:
                return cls(rounded, negative=False, ctx=ctx)
        else:
            return cls(rounded, ctx=ctx)

    def isnormal(self):
        return not (
            self.is_zero()
            or self.isinf
            or self.isnan
            or self.p <= 1
        )

    # in order to sink

    @staticmethod
    def _limiting_p(*args):
        p = None
        for arg in args:
            if arg.inexact and (p is None or arg.p < p):
                p = arg.p
        return p

    @staticmethod
    def _limiting_n(*args):
        n = None
        for arg in args:
            if arg.inexact and (n is None or arg.n > n):
                n = arg.n
        return n

    @staticmethod
    def _limiting_exactness(*args):
        for arg in args:
            if arg.inexact:
                return True
        return None

    def add(self, other, ctx=None):
        ctx = self._select_context(self, other, ctx=ctx)
        n = self._limiting_n(self, other)
        inexact = self._limiting_exactness(self, other)
        result = gmpmath.compute(OP.add, self, other, prec=ctx.p)
        return self._round_to_context(result, min_n=n, inexact=inexact, ctx=ctx, strict=True)

    def sub(self, other, ctx=None):
        ctx = self._select_context(self, other, ctx=ctx)
        n = self._limiting_n(self, other)
        inexact = self._limiting_exactness(self, other)
        result = gmpmath.compute(OP.sub, self, other, prec=ctx.p)
        return self._round_to_context(result, min_n=n, inexact=inexact, ctx=ctx, strict=True)

    def mul(self, other, ctx=None):
        ctx = self._select_context(self, other, ctx=ctx)
        p = self._limiting_p(self, other)
        inexact = self._limiting_exactness(self, other)
        result = gmpmath.compute(OP.mul, self, other, prec=ctx.p)
        return self._round_to_context(result, max_p=p, inexact=inexact, ctx=ctx, strict=True)

    def div(self, other, ctx=None):
        ctx = self._select_context(self, other, ctx=ctx)
        p = self._limiting_p(self, other)
        inexact = self._limiting_exactness(self, other)
        result = gmpmath.compute(OP.div, self, other, prec=ctx.p)
        return self._round_to_context(result, max_p=p, inexact=inexact, ctx=ctx, strict=True)

    def sqrt(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        p = self._limiting_p(self)
        # experimentally, it seems that precision increases for fractional exact powers
        if p is not None:
            p += 1
            p = min(p, ctx.p)
        inexact = self._limiting_exactness(self)
        result = gmpmath.compute(OP.sqrt, self, prec=ctx.p)
        return self._round_to_context(result, max_p=p, inexact=inexact, ctx=ctx, strict=True)

    def neg(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = type(self)(self, negative=(not self.negative))
        return self._round_to_context(result, ctx=ctx, strict=False)

    def fabs(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = type(self)(self, negative=False)
        return self._round_to_context(result, ctx=ctx, strict=False)



class Interpreter(interpreter.StandardInterpreter):
    dtype = SinkingPosit
    ctype = PositCtx

    @classmethod
    def arg_to_digital(cls, x, ctx):
        return cls.dtype(x, ctx=ctx)

    @classmethod
    def _eval_constant(cls, e, ctx):
        return cls.round_to_context(gmpmath.compute_constant(e.value, prec=ctx.nbits), ctx=ctx)

    # unfortunately, interpreting these values efficiently requries info from the context,
    # so it has to be implemented per interpreter...

    @classmethod
    def _eval_integer(cls, e, ctx):
        x = digital.Digital(m=e.i, exp=0, inexact=False)
        return cls.round_to_context(x, ctx=ctx)

    @classmethod
    def _eval_rational(cls, e, ctx):
        p = digital.Digital(m=e.p, exp=0, inexact=False)
        q = digital.Digital(m=e.q, exp=0, inexact=False)
        x = gmpmath.compute(OP.div, p, q, prec=ctx.nbits)
        return cls.round_to_context(x, ctx=ctx)

    @classmethod
    def _eval_digits(cls, e, ctx):
        x = gmpmath.compute_digits(e.m, e.e, e.b, prec=ctx.nbits)
        return cls.round_to_context(x, ctx=ctx)

    @classmethod
    def round_to_context(cls, x, ctx):
        """Not actually used???"""
        return cls.dtype._round_to_context(x, ctx=ctx, strict=False)
