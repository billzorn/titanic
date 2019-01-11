"""Emulated Posit arithmetic.
"""

from ..titanic import gmpmath
from ..titanic import digital

from ..titanic.integral import bitmask
from ..titanic.ops import RM, OP
from .evalctx import PositCtx
from . import mpnum
from . import interpreter


used_ctxs = {}
def posit_ctx(es, nbits):
    try:
        return used_ctxs[(es, nbits)]
    except KeyError:
        ctx = PositCtx(es=es, nbits=nbits)
        used_ctxs[(es, nbits)] = ctx
        return ctx

class Posit(mpnum.MPNum):

    _ctx : PositCtx = posit_ctx(4, 64)

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
            if kwargs:
                raise ValueError('cannot specify additional values {}'.format(repr(kwargs)))
            f = gmpmath.mpfr(x, ctx.p)
            unrounded = gmpmath.mpfr_to_digital(f)
            super().__init__(x=self._round_to_context(unrounded, ctx=ctx, strict=True))

        self._ctx = posit_ctx(ctx.es, ctx.nbits)

    def __repr__(self):
        return '{}(negative={}, c={}, exp={}, inexact={}, rc={}, isinf={}, isnan={}, ctx={})'.format(
            type(self).__name__, repr(self._negative), repr(self._c), repr(self._exp),
            repr(self._inexact), repr(self._rc), repr(self._isinf), repr(self._isnan), repr(self._ctx)
        )

    def __str__(self):
        return str(gmpmath.digital_to_mpfr(self))

    def __float__(self):
        return float(gmpmath.digital_to_mpfr(self))

    @classmethod
    def _select_context(cls, *args, ctx=None):
        if ctx is not None:
            return posit_ctx(ctx.es, ctx.nbits)
        else:
            nbits = max((f.ctx.nbits for f in args if isinstance(f, cls)))
            es = max((f.ctx.es for f in args if isinstance(f, cls) and f.ctx.nbits == nbits))
            return posit_ctx(es, nbits)

    @classmethod
    def _round_to_context(cls, unrounded, ctx=None, strict=False):
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

                print('->\t', new_exp, rc, exp_inexact)
                    
                # We want to round on the geometric mean of the two numbers,
                # but this is the same as rounding on the arithmetic mean of
                # the exponents.

                if half_bit > 0:
                    if low_bits > 0 or lost_sig_bits > 0 or unrounded.rc > 0:
                        # round the exponent up; remember it might be negative, but that's ok
                        print('UP')
                        new_exp += 1
                        rc = -1
                    elif unrounded.rc < 1:
                        # tie broken the other way
                        print('TIE BROKEN DOWN')
                        pass
                    elif new_exp & 1:
                        # hard coded rne
                        # TODO: not clear if this is actually what should happen
                        print('TIE UP')
                        new_exp += 1
                        rc = -1

                new_exp <<= offset
                rounded = digital.Digital(negative=unrounded.negative, c=1, exp=new_exp, inexact=exp_inexact, rc=rc)

            else:
                # we can represent the entire exponent, so only round the mantissa
                rounded = unrounded.round_m(max_p=sbits + 1, min_n=None, rm=RM.RNE, strict=strict)

        # Posits do not have a signed zero, and never round down to zero.

        if rounded.is_zero():
            if rounded.inexact:
                return cls(negative=rounded.negative, c=1, exp=ctx.emin, rc=-1, ctx=ctx)
            else:
                return cls(rounded, negative=False, ctx=ctx)
        else:
            return cls(rounded, ctx=ctx)

    # most operations come from mpnum

    def isnormal(self):
        return not (
            self.is_zero()
            or self.isinf
            or self.isnan
            or self.p <= 1
        )


class Interpreter(interpreter.StandardInterpreter):
    dtype = Posit
    ctype = PositCtx

    @classmethod
    def arg_to_digital(cls, x, ctx):
        return cls.dtype(x, ctx=ctx)

    @classmethod
    def _eval_constant(cls, e, ctx):
        return cls.round_to_context(gmpmath.compute_constant(e.value, prec=ctx.nbits), ctx=ctx)

    @classmethod
    def round_to_context(cls, x, ctx):
        """Not actually used???"""
        return cls.dtype._round_to_context(x, ctx=ctx, strict=False)








def arg_to_digital(x, ctx=posit_ctx(4, 64)):
    result = gmpmath.mpfr_to_digital(gmpmath.mpfr(x, ctx.nbits - ctx.es))
    return round_to_posit_ctx(result, ctx=ctx)


def digital_to_bits(x, ctx=posit_ctx(4, 64)):
    if ctx.nbits < 2 or ctx.es < 0:
        raise ValueError('format with nbits={}, es={} cannot be represented with posit bit pattern'.format(ctx.nbits, ctx.es))

    try:
        rounded = round_to_posit_ctx(x, ctx)
    except sinking.PrecisionError:
        rounded = round_to_posit_ctx(sinking.Sink(x, inexact=False), ctx)

    if rounded.isnan:
        return 1 << (ctx.nbits - 1)
    elif rounded.is_zero():
        return 0

    regime, e = divmod(x.e, ctx.u)

    if regime < 0:
        R = 1
        rbits = -regime + 1
    else:
        R = ((1 << (regime + 1)) - 1) << 1
        rbits = regime + 2

    sbits = ctx.nbits - 1 - rbits - ctx.es

    if sbits < -ctx.es:
        X = R >> -(ctx.es + sbits)
    elif sbits <= 0:
        X = (R << (ctx.es + sbits)) | (e >> -sbits)
    else:
        X = (R << (ctx.es + sbits)) | (e << sbits) | (rounded.c & bitmask(sbits))

    if rounded.negative:
        return -X & bitmask(ctx.nbits)
    else:
        return X


def show_bitpattern(x, ctx=posit_ctx(4, 64)):
    if isinstance(x, int):
        i = x
    elif isinstance(x, sinking.Sink):
        i = digital_to_bits(x, ctx=ctx)

    if i & (1 << (ctx.nbits - 1)) == 0:
        X = i
        sign = '+'
    else:
        X = -i & bitmask(ctx.nbits - 1)
        sign = '-'

    if X == 0:
        if sign == '+':
            return ('posit{:d}({:d}): zero {:0'+str(ctx.nbits - 1)+'b}').format(ctx.nbits, ctx.es, X)
        else:
            return ('posit{:d}({:d}): NaR {:0'+str(ctx.nbits - 1)+'b}').format(ctx.nbits, ctx.es, X)

    # detect the regime

    idx = ctx.nbits - 2
    r = (X >> idx) & 1

    while idx > 0 and (X >> (idx - 1) & 1) == r:
        idx -= 1

    # the regime extends one index past idx (or to idx if idx is 0)

    ebits = max(idx - 1, 0)
    rbits = ctx.nbits - 1 - ebits

    if ebits > ctx.es:
        sbits = ebits - ctx.es
        ebits = ctx.es
    else:
        sbits = 0

    if sbits > 0:
        return ('posit{:d}({:d}): {:s} {:0'+str(rbits)+'b} {:0'+str(ebits)+'b} (1) {:0'+str(sbits)+'b}').format(
            ctx.nbits, ctx.es, sign, X >> (ebits + sbits), (X >> sbits) & bitmask(ebits), X & bitmask(sbits),
        )
    elif ebits > 0:
        return ('posit{:d}({:d}): {:s} {:0'+str(rbits)+'b} {:0'+str(ebits)+'b}').format(
            ctx.nbits, ctx.es, sign, X >> ebits, X & bitmask(ebits),
        )
    else:
        return ('posit{:d}({:d}): {:s} {:0'+str(rbits)+'b}').format(
            ctx.nbits, ctx.es, sign, X,
        )
