"""Emulated IEEE 754 floating-point arithmetic.
"""

from ..titanic import gmpmath
from ..titanic import digital

from ..titanic.integral import bitmask
from ..titanic.ops import RM, OP
from .evalctx import IEEECtx
from . import mpnum
from . import interpreter


used_ctxs = {}
def ieee_ctx(es, nbits, rm=RM.RNE):
    try:
        return used_ctxs[(es, nbits, rm)]
    except KeyError:
        ctx = IEEECtx(es=es, nbits=nbits, rm=rm)
        used_ctxs[(es, nbits, rm)] = ctx
        return ctx


class Float(mpnum.MPNum):

    _ctx : IEEECtx = ieee_ctx(11, 64)

    @property
    def ctx(self):
        """The rounding context used to compute this value.
        If a computation takes place between two values, then
        it will either use a provided context (which will be recorded
        on the result) or the more precise of the parent contexts
        if none is provided.
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

        self._ctx = ieee_ctx(ctx.es, ctx.nbits, rm=ctx.rm)

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
            return ieee_ctx(ctx.es, ctx.nbits, rm=ctx.rm)
        else:
            es = max((f.ctx.es for f in args if isinstance(f, cls)))
            p = max((f.ctx.p for f in args if isinstance(f, cls)))
            return ieee_ctx(es, es + p)

    @classmethod
    def _round_to_context(cls, unrounded, ctx=None, strict=False):
        if ctx is None:
            if isinstance(unrounded, cls):
                ctx = unrounded.ctx
            else:
                raise ValueError('no context specified to round {}'.format(repr(unrounded)))

        if unrounded.isinf or unrounded.isnan:
            return cls(unrounded, ctx=ctx)

        magnitude = cls(unrounded, negative=False)
        if magnitude < ctx.fbound:
            return cls(unrounded.round_new(max_p=ctx.p, min_n=ctx.n, rm=ctx.rm, strict=strict), ctx=ctx)
            #return cls(unrounded.round_new(max_p=ctx.p, min_n=ctx.n, rm=ctx.rm, strict=strict), ctx=ctx)
        else:
            if magnitude > ctx.fbound or magnitude.rc >= 0:
                return cls(negative=unrounded.negative, isinf=True, ctx=ctx)
            else:
                return cls(unrounded.round_new(max_p=ctx.p, min_n=ctx.n, rm=ctx.rm, strict=strict), ctx=ctx)

    # most operations come from mpnum

    def isnormal(self):
        return not (
            self.is_zero()
            or self.isinf
            or self.isnan
            or self.e < self.ctx.emin
        )


class Interpreter(interpreter.StandardInterpreter):
    dtype = Float
    ctype = IEEECtx

    @classmethod
    def arg_to_digital(cls, x, ctx):
        return cls.dtype(x, ctx=ctx)

    @classmethod
    def _eval_constant(cls, e, ctx):
        return cls.round_to_context(gmpmath.compute_constant(e.value, prec=ctx.p), ctx=ctx)

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
        x = gmpmath.compute(OP.div, p, q, prec=ctx.p)
        return cls.round_to_context(x, ctx=ctx)

    @classmethod
    def _eval_digits(cls, e, ctx):
        x = gmpmath.compute_digits(e.m, e.e, e.b, prec=ctx.p)
        return cls.round_to_context(x, ctx=ctx)

    @classmethod
    def round_to_context(cls, x, ctx):
        """Not actually used?"""
        return cls.dtype._round_to_context(x, ctx=ctx, strict=False)



def digital_to_bits(x, ctx=ieee_ctx(11, 53)):
    if ctx.p < 2 or ctx.w < 2:
        raise ValueError('format with w={}, p={} cannot be represented with IEEE 754 bit pattern'.format(ctx.w, ctx.p))

    try:
        rounded = round_to_ieee_ctx(x, ctx)
    except sinking.PrecisionError:
        rounded = round_to_ieee_ctx(sinking.Sink(x, inexact=False), ctx)

    pbits = ctx.p - 1

    if rounded.negative:
        S = 1
    else:
        S = 0

    if rounded.isnan:
        # canonical NaN
        return (0 << (ctx.w + pbits)) | (bitmask(ctx.w) << pbits) | (1 << (pbits - 1))
    elif rounded.isinf:
        return (S << (ctx.w + pbits)) | (bitmask(ctx.w) << pbits) # | 0
    elif rounded.is_zero():
        return (S << (ctx.w + pbits)) # | (0 << pbits) | 0

    c = rounded.c
    cbits = rounded.p
    e = rounded.e

    if e < ctx.emin:
        # subnormal
        lz = (ctx.emin - 1) - e
        if lz > pbits or (lz == pbits and cbits > 0):
            raise ValueError('exponent out of range: {}'.format(e))
        elif lz + cbits > pbits:
            raise ValueError('too much precision: given {}, can represent {}'.format(cbits, pbits - lz))
        E = 0
        C = c << (lz - (pbits - cbits))
    elif e <= ctx.emax:
        # normal
        if cbits > ctx.p:
            raise ValueError('too much precision: given {}, can represent {}'.format(cbits, ctx.p))
        elif cbits < ctx.p:
            raise ValueError('too little precision: given {}, can represent {}'.format(cbits, ctx.p))
        E = e + ctx.emax
        C = (c << (ctx.p - cbits)) & bitmask(pbits)
    else:
        # overflow
        raise ValueError('exponent out of range: {}'.format(e))

    return (S << (ctx.w + pbits)) | (E << pbits) | C


def bits_to_digital(i, ctx=ieee_ctx(11, 53)):
    pbits = ctx.p - 1

    S = (i >> (ctx.w + pbits)) & bitmask(1)
    E = (i >> pbits) & bitmask(ctx.w)
    C = i & bitmask(pbits)

    negative = (S == 1)
    e = E - ctx.emax

    if E == 0:
        # subnormal
        c = C
        exp = -ctx.emax - pbits + 1
    elif e <= ctx.emax:
        # normal
        c = C | (1 << pbits)
        exp = e - pbits
    else:
        # nonreal
        if C == 0:
            return sinking.Sink(negative=negative, c=0, exp=0, inf=True, rc=0)
        else:
            return sinking.Sink(negative=False, c=0, exp=0, nan=True, rc=0)

    # unfortunately any rc / exactness information is lost
    return sinking.Sink(negative=negative, c=c, exp=exp, inexact=False, rc=0)


def show_bitpattern(x, ctx=ieee_ctx(11, 53)):
    print(x)

    if isinstance(x, int):
        i = x
    elif isinstance(x, sinking.Sink):
        i = digital_to_bits(x, ctx=ctx)

    S = i >> (ctx.w + ctx.p - 1)
    E = (i >> (ctx.p - 1)) & bitmask(ctx.w)
    C = i & bitmask(ctx.p - 1)
    if E == 0 or E == bitmask(ctx.w):
        hidden = 0
    else:
        hidden = 1

    return ('float{:d}({:d},{:d}): {:01b} {:0'+str(ctx.w)+'b} ({:01b}) {:0'+str(ctx.p-1)+'b}').format(
        ctx.w + ctx.p, ctx.w, ctx.p, S, E, hidden, C,
    )


# import numpy as np
# import sys
# def bits_to_numpy(i, nbytes=8, dtype=np.float64):
#     return np.frombuffer(
#         i.to_bytes(nbytes, sys.byteorder),
#         dtype=dtype, count=1, offset=0,
#     )[0]
