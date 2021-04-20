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

    _ctx : PositCtx = posit_ctx(3, 64)

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
                        bits = digital_to_bits(rounded, ctx)
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
                    if cbits == 1:
                        half_bit = 0
                        low_bits = 0
                    else:
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
                        bits = digital_to_bits(rounded, ctx)
                        if bits & 1:
                            new_exp += 1
                            rc = -1

                rounded = digital.Digital(negative=unrounded.negative, c=1, exp=new_exp, inexact=exp_inexact, rc=rc)


            else:
                #print('normal round')
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

    def arg_to_digital(self, x, ctx):
        return self.dtype(x, ctx=ctx)

    def _eval_constant(self, e, ctx):
        try:
            return None, self.constants[e.value]
        except KeyError:
            return None, self.round_to_context(gmpmath.compute_constant(e.value, prec=ctx.nbits), ctx=ctx)

    # unfortunately, interpreting these values efficiently requries info from the context,
    # so it has to be implemented per interpreter...

    def _eval_integer(self, e, ctx):
        x = digital.Digital(m=e.i, exp=0, inexact=False)
        return None, self.round_to_context(x, ctx=ctx)

    def _eval_rational(self, e, ctx):
        p = digital.Digital(m=e.p, exp=0, inexact=False)
        q = digital.Digital(m=e.q, exp=0, inexact=False)
        x = gmpmath.compute(OP.div, p, q, prec=ctx.nbits)
        return None, self.round_to_context(x, ctx=ctx)

    def _eval_digits(self, e, ctx):
        x = gmpmath.compute_digits(e.m, e.e, e.b, prec=ctx.nbits)
        return None, self.round_to_context(x, ctx=ctx)

    def round_to_context(self, x, ctx):
        """Not actually used???"""
        return self.dtype._round_to_context(x, ctx=ctx, strict=False)








def arg_to_digital(x, ctx=posit_ctx(4, 64)):
    result = gmpmath.mpfr_to_digital(gmpmath.mpfr(x, ctx.nbits - ctx.es))
    return round_to_posit_ctx(result, ctx=ctx)


def digital_to_bits(x, ctx=None):
    if ctx == None and not isinstance(x, Posit):
        raise ValueError('must provide a format to convert {} to bits'.format(repr(x)))

    if ctx is not None:
        rounded = x
    else:
        rounded = Interpreter.round_to_context(x, ctx)
        ctx = rounded.ctx

    if ctx.nbits < 2 or ctx.es < 0:
        raise ValueError('format with nbits={}, es={} cannot be represented with posit bit pattern'.format(ctx.nbits, ctx.es))

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


def bits_to_digital(i, ctx=posit_ctx(4, 64)):
    if i & (1 << (ctx.nbits - 1)) == 0:
        X = i
        negative = False
    else:
        X = -i & bitmask(ctx.nbits - 1)
        negative = True

    if X == 0:
        if negative:
            return Posit(isnan=True, negative=negative, inexact=False, rounded=False, rc=0, ctx=ctx)
        else:
            return Posit(c=0, exp=0, negative=negative, inexact=False, rounded=False, rc=0, ctx=ctx)

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

    # pull out bitfields
    regime = X >> (ebits + sbits)
    exponent = (X >> sbits) & bitmask(ebits)
    significand = (X & bitmask(sbits)) | (1 << sbits)

    #print('rbits={}, ebits={}, sbits={}'.format(rbits, ebits, sbits))
    #print('regime={}, exponent={}, significand={}'.format(regime, exponent, significand))

    # convert regime
    if regime == 1:
        regime = regime - rbits
    else:
        regime = rbits - (4 - (regime & 3))

    # fix up exponent
    if ebits < ctx.es:
        exponent <<= (ctx.es - ebits)

    #print(regime, exponent)

    return Posit(c=significand, e=(ctx.u*regime) + exponent, negative=negative, inexact=False, rounded=False, rc=0, ctx=ctx)


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



# import sfpy
# import random
# ctx8 = posit_ctx(0, 8)
# ctx16 = posit_ctx(1, 16)
# ctx32 = posit_ctx(2, 32)

# def test_posit_bits(ctx=ctx8, nbits=8, sfptype=sfpy.Posit8, tests=None):
#     if tests is None:
#         rng = range(1<<nbits)
#     else:
#         lim = (1<<nbits) - 1
#         rng = [random.randint(0, lim) for _ in range(tests)]

#     for i in rng:
#         sfp = sfptype(i)
#         p = Posit(float(sfp), ctx)

#         if not digital_to_bits(p) == sfp.bits:
#             print(float(sfp), 'expected', sfp.bits, ': got', digital_to_bits(p))

#         #print('----')
#         p2 = bits_to_digital(i, ctx)
#         if not float(p2) == float(sfp):
#             print(i, 'expected', sfp, ': got', str(p2))
#             print(show_bitpattern(i, ctx))
#             trueregime, truee = divmod(p.e, ctx.u)
#             print('should have regime={}, e={}'.format(trueregime, truee))



# def bad_rounding():
#     cases16 = [
#         -25165824.0,
#         -2.2351741790771484e-08,
#         -7.450580596923828e-09,
#         25165824.0,
#         2.2351741790771484e-08,
#         7.450580596923828e-09,
#         -67108863.99999999,
#         -67108863.999999985,
#         -50331648.00000001,
#         -50331648.0,
#         -33554431.999999996,
#         -33554431.999999993,
#         -25165824.000000004,
#         -5.960464477539062e-08,
#         -5.960464477539061e-08,
#         -4.4703483581542975e-08,
#         -4.470348358154297e-08,
#         -2.980232238769531e-08,
#         -2.9802322387695306e-08,
#         -2.2351741790771488e-08,
#         2.2351741790771488e-08,
#         2.9802322387695306e-08,
#         2.980232238769531e-08,
#         4.470348358154297e-08,
#         4.4703483581542975e-08,
#         5.960464477539061e-08,
#         5.960464477539062e-08,
#         25165824.000000004,
#         33554431.999999993,
#         33554431.999999996,
#         50331648.0,
#         50331648.00000001,
#         67108863.999999985,
#         67108863.99999999,
#     ]

#     for case in cases16:

#         print('----')
#         sfp = sfpy.Posit16(case)
#         p = Posit(case, ctx=ctx16)

#         if not float(sfp) == float(p):
#             print(case, 'expected', sfp, ': got', str(p))
