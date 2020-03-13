"""Sinking-point for IEEE 754-like arithmetic.
"""

from ..titanic import gmpmath
from ..titanic import digital

from ..titanic.integral import bitmask
from ..titanic.ops import RM, OP
from .evalctx import IEEECtx
from . import interpreter
from . import ieee754

class Sink(digital.Digital):
    _ctx : IEEECtx = ieee754.ieee_ctx(11, 64)

    @property
    def ctx(self):
        """The rounding context used to compute this value.
        If a computation takes place between two values, then
        it will either use a provided context (which will be recorded
        on the result) or the more precise of the parent contexts
        if none is provided.
        The context is a limit on the precision to store; actual
        results may have significantly less precision, depending
        on the sinking-point rounding rules.
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

        self._ctx = ieee754.ieee_ctx(ctx.es, ctx.nbits)

    def __repr__(self):
        return '{}(negative={}, c={}, exp={}, inexact={}, rc={}, isinf={}, isnan={}, ctx={})'.format(
            type(self).__name__, repr(self._negative), repr(self._c), repr(self._exp),
            repr(self._inexact), repr(self._rc), repr(self._isinf), repr(self._isnan), repr(self._ctx)
        )

    def __str__(self):
        if self.isnan:
            return 'nan'
        elif self.isinf:
            if self.negative:
                return '-inf'
            else:
                return 'inf'

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
            return ieee754.ieee_ctx(ctx.es, ctx.p)
        else:
            w = max((f.ctx.es for f in args if isinstance(f, cls)))
            p = max((f.ctx.p for f in args if isinstance(f, cls)))
            return ieee754.ieee_ctx(w, p)

    @classmethod
    def _round_to_context(cls, unrounded, max_p=None, min_n=None, inexact=None, ctx=None, strict=False):
        if ctx is None:
            if hasattr(unrounded, 'ctx'):
                ctx = unrounded.ctx
            else:
                raise ValueError('no context specified to round {}'.format(repr(unrounded)))

        if ctx.rm != RM.RNE:
            raise ValueError('unimplemented rounding mode {}'.format(repr(rm)))

        if unrounded.isinf or unrounded.isnan:
            return cls(unrounded, ctx=ctx)

        magnitude = cls(unrounded, negative=False)
        if magnitude > ctx.fbound or (magnitude == ctx.fbound and magnitude.rc >= 0):
            return cls(negative=unrounded.negative, isinf=True, ctx=ctx)
        else:
            # get p and n used for rounding, limited by the context
            if max_p is None:
                p = ctx.p
            else:
                p = min(ctx.p, max_p)
            if min_n is None:
                n = ctx.n
            else:
                n = max(ctx.n, min_n)

            # round with the chosed n and p
            rounded = cls(unrounded.round_m(max_p=p, min_n=n, rm=ctx.rm, strict=strict), ctx=ctx)

            # In gmp/mpfr, the exactness of the result is only based on what happened during
            # the immediate computation, assuming that all inputs are exact.
            # Sinking-point tracks exactness through a computation, so if we know that the inputs
            # were inexact, we may need to mark the result as inexact, even though the computation
            # itself could be performed exactly.
            #
            # Note that we will lose a little bit of information here about the rc.
            # An exactly computed result which we know is inexact will still have rc=0.
            # This is fine - it just means we don't know which way the value was rounded.
            # In some cases, we might be able to figure it out (i.e. addition of opposite signs
            # with opposite result codes) but in general this is very hard.
            #
            # The inexact argument is an override; if it is False, the result will
            # be exact even if the computation seemed to cause rounding.
            if inexact is not None and inexact != rounded.inexact:
                return cls(rounded, inexact=inexact)
            else:
                return rounded

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

    # temporary hack
    def sin(self, ctx=None):
        ctx = self._select_context(self, ctx=ctx)
        result = gmpmath.compute(OP.sin, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=False)


class Interpreter(interpreter.StandardInterpreter):
    dtype = Sink
    ctype = IEEECtx

    def arg_to_digital(self, x, ctx):
        return self.dtype(x, ctx=ctx)

    def _eval_constant(self, e, ctx):
        try:
            return None, self.constants[e.value]
        except KeyError:
            return None, self.round_to_context(gmpmath.compute_constant(e.value, prec=ctx.p), ctx=ctx)

    # unfortunately, interpreting these values efficiently requries info from the context,
    # so it has to be implemented per interpreter...

    def _eval_integer(self, e, ctx):
        x = digital.Digital(m=e.i, exp=0, inexact=False)
        return None, self.round_to_context(x, ctx=ctx)

    def _eval_rational(self, e, ctx):
        p = digital.Digital(m=e.p, exp=0, inexact=False)
        q = digital.Digital(m=e.q, exp=0, inexact=False)
        x = gmpmath.compute(OP.div, p, q, prec=ctx.p)
        return None, self.round_to_context(x, ctx=ctx)

    def _eval_digits(self, e, ctx):
        x = gmpmath.compute_digits(e.m, e.e, e.b, prec=ctx.p)
        return None, self.round_to_context(x, ctx=ctx)

    def round_to_context(self, x, ctx):
        """Not actually used?"""
        return self.dtype._round_to_context(x, ctx=ctx, strict=False)
