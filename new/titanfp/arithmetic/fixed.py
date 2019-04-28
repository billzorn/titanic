"""Emulated fixed-point arithmetic, also useful for quires.
"""

from ..titanic import digital
from ..titanic import gmpmath

from .evalctx import FixedCtx
from . import mpnum
from . import interpreter


used_ctxs = {}
def fixed_ctx(p, n):
    try:
        return used_ctxs[(p, n)]
    except KeyError:
        ctx = FixedCtx(p=p, n=n)
        used_ctxs[(p, n)] = ctx
        return ctx


class Fixed(mpnum.MPNum):

    _ctx : FixedCtx = fixed_ctx(64, -33)

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
            return super().is_identical_to(other) and self.ctx.p == other.ctx.p and self.ctx.n == other.ctx.n
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

        self._ctx = fixed_ctx(ctx.p, ctx.n)

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
            return fixed_ctx(ctx.p, ctx.n)
        else:
            n = min((f.ctx.n for f in args if isinstance(f, cls)))
            maxbit = max((f.ctx.n + f.ctx.p for f in args if isinstance(f, cls)))
            p = maxbit - n
            return fixed_ctx(p, n)

    @classmethod
    def _round_to_context(cls, unrounded, ctx=None, strict=False):
        if ctx is None:
            if isinstance(unrounded, cls):
                ctx = unrounded.ctx
            else:
                raise ValueError('no context specified to round {}'.format(repr(unrounded)))

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
