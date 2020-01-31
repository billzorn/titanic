"""Emulated fixed-point arithmetic, also useful for quires.
"""

from ..titanic import digital
from ..titanic import gmpmath
from ..titanic.ops import RM, OF

from .evalctx import FixedCtx
from . import mpnum
from . import interpreter


used_ctxs = {}
def fixed_ctx(scale, nbits, rm=RM.RTN, of=OF.INFINITY):
    try:
        return used_ctxs[(scale, nbits, rm, of)]
    except KeyError:
        ctx = FixedCtx(scale=scale, nbits=nbits, rm=rm, of=of)
        used_ctxs[(scale, nbits, rm, of)] = ctx
        return ctx


class Fixed(mpnum.MPNum):

    _ctx : FixedCtx = fixed_ctx(0, 64)

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
            return super().is_identical_to(other) and self.ctx.scale == other.ctx.scale and self.ctx.nbits == other.ctx.nbits
        else:
            return super().is_identical_to(other)

    def __init__(self, x=None, ctx=None, **kwargs):
        if ctx is None:
            ctx = type(self)._ctx

        if x is None or isinstance(x, digital.Digital):
            super().__init__(x=x, **kwargs)
        else:
            if kwargs:
                raise ValueE('cannot specify additional values {}'.format(repr(kwargs)))
            f = gmpmath.mpfr(x, ctx.p)
            unrounded = gmpmath.mpfr_to_digital(f)
            super().__init__(x=self._round_to_context(unrounded, ctx=ctx, strict=True))

        self._ctx = fixed_ctx(ctx.scale, ctx.nbits, rm=ctx.rm, of=ctx.of)

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
            return fixed_ctx(ctx.scale, ctx.nbits, rm=ctx.rm, of=ctx.of)
        else:
            n = min((f.ctx.n for f in args if isinstance(f, cls)))
            maxbit = max((f.ctx.n + f.ctx.p for f in args if isinstance(f, cls)))
            nbits = maxbit - n
            return fixed_ctx(n + 1, nbits)

    @classmethod
    def _round_to_context(cls, unrounded, ctx=None, strict=False):
        if ctx is None:
            if isinstance(unrounded, cls):
                ctx = unrounded.ctx
            else:
                raise ValueError('no context specified to round {}'.format(repr(unrounded)))

        if  ctx.of != OF.INFINITY:
            raise ValueError('unsupported overflow mode {}'.format(str(ctx.of)))

        if unrounded.isinf or unrounded.isnan:
            return cls(unrounded, ctx=ctx)

        # do a size check now, to avoid attempting to round to more digits than we have
        if unrounded.e > ctx.n + ctx.p:
            return cls(unrounded, isinf=True)

        rounded = unrounded.round_new(min_n=ctx.n, rm=ctx.rm, strict=strict)

        # fix up rc, to be compatible with old rounding code
        if rounded.rounded:
            if rounded.interval_down:
                rounded = cls(rounded, rc=-1)
            else:
                rounded = cls(rounded, rc=1)
        else:
            rounded = cls(rounded, rc=0)

        if rounded.c.bit_length() > ctx.p:
            rounded = cls(rounded, isinf=True)

        return cls(rounded, ctx=ctx)


class Interpreter(interpreter.StandardInterpreter):
    dtype = Fixed
    ctype = FixedCtx

    def arg_to_digital(self, x, ctx):
        return self.dtype(x, ctx=ctx)

    def _eval_constant(self, e, ctx):
        return self.round_to_context(gmpmath.compute_constant(e.value, prec=ctx.p), ctx=ctx)

    # unfortunately, interpreting these values efficiently requries info from the context,
    # so it has to be implemented per interpreter...

    def _eval_integer(self, e, ctx):
        x = digital.Digital(m=e.i, exp=0, inexact=False)
        return self.round_to_context(x, ctx=ctx)

    def _eval_rational(self, e, ctx):
        p = digital.Digital(m=e.p, exp=0, inexact=False)
        q = digital.Digital(m=e.q, exp=0, inexact=False)
        x = gmpmath.compute(OP.div, p, q, prec=ctx.p)
        return self.round_to_context(x, ctx=ctx)

    def _eval_digits(self, e, ctx):
        x = gmpmath.compute_digits(e.m, e.e, e.b, prec=ctx.p)
        return self.round_to_context(x, ctx=ctx)

    def round_to_context(self, x, ctx):
        """Not actually used?"""
        return self.dtype._round_to_context(x, ctx=ctx, strict=False)
