"""Interval arithmetic based on Rival
Original implementation: https://github.com/herbie-fp/rival
"""

from enum import IntEnum, unique
from re import S
from unicodedata import digit

from numpy import isin

from titanfp.titanic import gmpmath
from . import interpreter
from . import ieee754

from ..titanic import digital
from ..titanic.ops import OP, RM

@unique
class IntervalSign(IntEnum):
    """Classification of an `Interval` by the sign of the endpoints.
    See `Interval.classify()` for details."""
    STRICTLY_NEGATIVE = -2,
    NEGATIVE          = -1,
    CONTAINS_ZERO     = 0,
    POSITIVE          = 1,
    STRICTLY_POSITIVE = 2,


class Interval(object):
    """An interval as originally implemented in the Rival interval arithmetic library.
    Original implementation at https://github.com/herbie-fp/rival written by
    Pavel Panchekha and Oliver Flatt. The interval uses MPFR floating-point value bounds,
    immovability flags to signal a fixed endpoint due to overflow, and error flags to propagate
    partial or complete domain errors.
    """

    # represents the real number interval [_lo, _hi]
    # (using ieee754.Float since this makes the most sense)
    _lo: ieee754.Float = ieee754.Float(digital.Digital(negative=True, isinf=True), ctx=ieee754.ieee_ctx(11, 64))
    _hi: ieee754.Float = ieee754.Float(digital.Digital(negative=False, isinf=True), ctx=ieee754.ieee_ctx(11, 64))

    # immovability flag
    # from the original Rival implementation:
    #
    #      "Intervals may shrink (though they cannot grow) when computed at a higher precision.
    #   However, in some cases it is known this will not occur, largely due to overflow.
    #   In those cases, the interval is marked fixed, or immovable."
    #
    _lo_isfixed: bool = False
    _hi_isfixed: bool = False

    # domain error, e.g. sqrt([-1, 1])
    _err: bool = False

    # invalid interval, e.g. sqrt([-2, -1])
    _invalid: bool = False

    # precision context
    _ctx : ieee754.IEEECtx = ieee754.ieee_ctx(11, 64)

    # the internal state is not directly visible: expose it with properties

    @property
    def lo(self):
        """The lower bound.
        """
        return self._lo

    @property
    def hi(self):
        """The lower bound.
        """
        return self._hi

    @property
    def lo_isfixed(self):
        """Is the lower bound immovable? 
        
        From the original Rival implementation:
        "Intervals may shrink (though they cannot grow) when computed at a higher precision.
        However, in some cases it is known this will not occur, largely due to overflow.
        In those cases, the interval is marked fixed, or immovable."
        """
        return self._lo_isfixed

    @property
    def hi_isfixed(self):
        """Is the upper bound immovable? 
        
        From the original Rival implementation:
        "Intervals may shrink (though they cannot grow) when computed at a higher precision.
        However, in some cases it is known this will not occur, largely due to overflow.
        In those cases, the interval is marked fixed, or immovable."
        """
        return self._hi_isfixed

    @property
    def err(self):
        """Did a (partial) domain error occur?
        Not to be confused with `invalid`    
        """
        return self._err

    @property
    def invalid(self):
        """Did a complete domain error occur?       
        """
        return self._invalid

    @property
    def ctx(self):
        """The rounding context used to compute this value.
        If a computation takes place between two values, then
        it will either use a provided context (which will be recorded
        on the result) or the more precise of the parent contexts
        if none is provided.
        """
        return self._ctx

    def __init__(self,
                 x=None,
                 lo=None,
                 hi=None,
                 ctx=None,
                 lo_isfixed=False,
                 hi_isfixed=False,
                 err=False,
                 invalid=False):
        """Creates a new interval. The first argument `x` is either an interval
        to clone and update, or a backing type to construct the smallest interval
        according to the format specified by `ctx`. The arguments `lo` and `hi`
        construts the interval `[lo, hi]`. If none of these arguments are provided,
        the interval is initialized to the real number line `[-inf, inf]`.
        All other fields can be optionally specified."""

        # _ctx
        if ctx is not None:
            self._ctx = ctx
        elif x is not None:
            if isinstance(x, Interval):
                self._ctx = x._ctx
            else:
                self._ctx = type(self)._ctx
        else:
            self._ctx = type(self)._ctx
        
        # rounding contexts for endpoints
        lo_ctx = ieee754.ieee_ctx(es=self._ctx.es, nbits=self._ctx.nbits, rm=RM.RTN)
        hi_ctx = ieee754.ieee_ctx(es=self._ctx.es, nbits=self._ctx.nbits, rm=RM.RTP)

        # _lo and _hi
        if x is not None:
            if lo is not None or hi is not None:
                raise ValueError('cannot specify both x={} and [lo={}, hi={}]'.format(repr(x), repr(lo), repr(hi)))
            if isinstance(x, type(self)):
                self._lo = ieee754.Float._round_to_context(x._lo, ctx=lo_ctx)
                self._hi = ieee754.Float._round_to_context(x._hi, ctx=lo_ctx)
            elif isinstance(x, ieee754.Float):
                self._lo = ieee754.Float._round_to_context(x, ctx=lo_ctx)
                self._hi = ieee754.Float._round_to_context(x, ctx=hi_ctx)
            else:
                self._lo = ieee754.Float(x=x, ctx=lo_ctx)
                self._hi = ieee754.Float(x=x, ctx=hi_ctx)
        elif lo is not None:
            if hi is None:
                raise ValueError('must specify both lo={} and hi={} together'.format(repr(lo), repr(hi)))
            if x is not None:
                raise ValueError('cannot specify both x={} and [lo={}, hi={}]'.format(repr(x), repr(lo), repr(hi)))
            self._lo = ieee754.Float(x=lo, ctx=lo_ctx)
            self._hi = ieee754.Float(x=hi, ctx=hi_ctx)
            if self._lo > self._hi:
                raise ValueError('invalid interval: lo={}, hi={}'.format(self._lo, self._hi))                
        else:
            self._lo = type(self)._lo
            self._hi = type(self)._hi

        # _lo_isfixed
        if lo_isfixed is not None:
            self._lo_isfixed = lo_isfixed
        elif x is not None:
            if isinstance(x, Interval):
                self._lo_isfixed = x._lo_isfixed
            else:
                self._lo_isfixed = not self._lo.inexact
        else:
            self._lo_isfixed = type(self)._lo_isfixed

        # _hi_isfixed
        if hi_isfixed is not None:
            self._hi_isfixed = hi_isfixed
        elif x is not None:
            if isinstance(x, Interval):
                self._hi_isfixed = x._hi_isfixed
            else:
                self._hi_isfixed = not self._hi.inexact
        else:
            self._hi_isfixed = type(self)._hi_isfixed

        # _err
        if err:
            self._err = err
        elif x is not None:
            if isinstance(x, Interval):
                self._err = x._err
            else:
                self._err = False
        else:
            self._err = type(self)._err

        # _invalid
        if invalid:
            self._invalid = invalid
        elif x is not None:
            if isinstance(x, Interval):
                self._invalid = x._invalid
            else:
                self._invalid = self._lo.isnan or self._hi.isnan
        else:
            self._invalid = type(self)._invalid

    def __repr__(self):
        return '{}(lo={}, hi={}, lo_isfixed={}, hi_isfixed={}, err={}, invalid={})'.format(
            type(self).__name__, repr(self._lo), repr(self._hi), repr(self._lo_isfixed),
            repr(self._hi_isfixed), repr(self._err), repr(self._invalid)
        )

    def __str__(self):
        if self._invalid:
            return '[nan, nan]'
        else:
            return '[{}, {}]'.format(str(self._lo), str(self._hi))

    # (visible) utility functions

    def is_point(self) -> bool:
        """"Is the interval a "singleton" interval, e.g. [a, a]?"""
        return self._lo == self._hi

    def contains(self, x) -> bool:
        """Does the interval contain `x`?"""
        return self._lo <= x and x <= self._hi

    def classify(self, strict=False) -> IntervalSign:
        """Classifies this interval by the sign of the endpoints.
        By default, this interval can be `NEGATIVE`, `CONTAINS_ZERO`, or `POSITIVE`
        where intervals with 0 as an endpoint are either `NEGATIVE` and `POSITIVE`.
        If `strict` is `True`, this interval can be `STRICTLY_POSITIVE`, `CONTAINS_ZERO`
        or `STRICTLY_NEGATIVE` where intervals with 0 as an endpoint are classified
        as `CONTAINS_ZERO`.
        """
        zero = digital.Digital(m=0, exp=0)
        if strict:
            if self._hi < zero:
                return IntervalSign.STRICTLY_NEGATIVE
            elif self._lo > zero:
                return IntervalSign.STRICTLY_POSITIVE
            else:
                return IntervalSign.CONTAINS_ZERO
        else:
            if self._hi <= zero:
                return IntervalSign.NEGATIVE
            elif self._lo >= zero:
                return IntervalSign.POSITIVE
            else:
                return IntervalSign.CONTAINS_ZERO

    def union(self, other):
        """Returns the union of this interval and another."""
        if self.invalid or other.invalid:
            return Interval(invalid=True)
        lo, lo_isfixed = self._lo_endpoint() if self._lo < other._lo else other._lo_endpoint()
        hi, hi_isfixed = self._hi_endpoint() if self._hi < other._hi else other._hi_endpoint()
        err = self._err or other._err
        return Interval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err)

    def clamp(self, lo, hi, err=False):
        """Returns a new interval that is clamped between `lo` and `hi`.
        If `err` is `True` and the current interval lies partially outside `[lo, hi]`,
        the `err` field of the new interval will be `true`.
        If `err` is `True` and the current interval lies completely outside `[lo, hi]`,
        the `invalid` field of the new interval will be `true`."""
        if lo > hi:
            raise ValueError('invalid clamp bounds: lo={}, hi={}'.format(self._lo, self._hi))

        # not a valid interval
        if self._invalid:
            return Interval(invalid=True)

        # below
        if self._hi < lo:
            if err:
                return Interval(invalid=True)
            else:
                return Interval(lo=lo, hi=lo, ctx=self._ctx)

        # above
        if self._lo > hi:
            if err:
                return Interval(invalid=True)
            else:
                return Interval(lo=hi, hi=hi, ctx=self._ctx)

        if self._lo < lo:
            if self._hi > hi:       # partially below and above
                return Interval(lo=lo, hi=hi, err=err, ctx=self._ctx)
            else:                   # partially below
                return Interval(lo=lo, hi=self._hi, err=err, ctx=self._ctx)
        elif self._hi > hi:         # partially above
            return Interval(lo=self._lo, hi=hi, err=err, ctx=self._ctx)
        else:                       # competely inside
            return Interval(x=self)

    # utility funtions

    def _lo_endpoint(self):
        return (self._lo, self._lo_endpoint)

    def _hi_endpoint(self):
        return (self._hi, self._hi_endpoint)

    def _select_context(self, *args, ctx=None):
        if ctx is None:
            es = max((ival._ctx.es for ival in args))
            p = max((ival._ctx.p for ival in args))
            ctx = ieee754.ieee_ctx(es, es + p)
        lo_ctx = ieee754.ieee_ctx(es=ctx.es, nbits=ctx.nbits, rm=RM.RTN)
        hi_ctx = ieee754.ieee_ctx(es=ctx.es, nbits=ctx.nbits, rm=RM.RTP)
        return ctx, lo_ctx, hi_ctx

    # simple endpoint computation that propagates immovability
    def _epfn(op, *eps, ctx):
        args = []
        args_fixed = False
        for ival, isfixed in eps:
            args.append(ival)
            args_fixed = args_fixed and isfixed

        result = gmpmath.compute(op, *args, prec=ctx.p)
        rounded = ieee754.Float._round_to_context(result, ctx=ctx)
        isfixed = args_fixed and not rounded.inexact
        return rounded, isfixed

    # endpoint computation for `add`, `sub`, `hypot`
    def _eplinear(op, a_ep, b_ep, ctx):
        a, a_isfixed = a_ep
        b, b_isfixed = b_ep
        result = gmpmath.compute(op, a, b, prec=ctx.p)
        rounded = ieee754.Float._round_to_context(result, ctx=ctx)
        isfixed = (a_isfixed and b_isfixed and not rounded.inexact) or \
                  (a_isfixed and a.isinf) or \
                  (b_isfixed and b.isinf)
        return rounded, isfixed

    # endpoint computation for `mul`
    def _epmul(op, a_ep, b_ep, aclass, bclass, ctx):
        a, a_isfixed, = a_ep
        b, b_isfixed, = b_ep
        result = gmpmath.compute(op, a, b, prec=ctx.p)
        rounded = ieee754.Float._round_to_context(result, ctx=ctx)
        isfixed = (a_isfixed and b_isfixed and not rounded.inexact) or \
                  (a_isfixed and a.is_zero() and not a.isinf) or \
                  (a_isfixed and a.isinf and bclass != IntervalSign.CONTAINS_ZERO) or \
                  (b_isfixed and b.is_zero() and not b.isinf) or \
                  (b_isfixed and b.isinf and aclass != IntervalSign.CONTAINS_ZERO)
        return rounded, isfixed

    # endpoint computation for `div`
    def _epdiv(op, a_ep, b_ep, aclass, ctx):
        a, a_isfixed, = a_ep
        b, b_isfixed, = b_ep
        result = gmpmath.compute(op, a, b, prec=ctx.p)
        rounded = ieee754.Float._round_to_context(result, ctx=ctx)
        isfixed = (a_isfixed and b_isfixed and not rounded.inexact) or \
                  (a_isfixed and a.is_zero() and not a.isinf) or \
                  (a_isfixed and a.isinf) or \
                  (b_isfixed and b.is_zero() and not b.isinf) or \
                  (b_isfixed and b.isinf and aclass != IntervalSign.CONTAINS_ZERO)
        return rounded, isfixed

    # helper for multiplication
    def _multiply(a, b, c, d, xclass, yclass, err, ctxs):
        ctx, lo_ctx, hi_ctx = ctxs
        lo, lo_isfixed = Interval._epmul(OP.mul, a, b, xclass, yclass, lo_ctx)
        hi, hi_isfixed = Interval._epmul(OP.mul, c, d, xclass, yclass, hi_ctx)
        return Interval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err, ctx=ctx)

    # helper for division
    def _divide(a, b, c, d, xclass, err, ctxs):
        ctx, lo_ctx, hi_ctx = ctxs
        lo, lo_isfixed = Interval._epdiv(OP.div, a, b, xclass, lo_ctx)
        hi, hi_isfixed = Interval._epdiv(OP.div, c, d, xclass, hi_ctx)
        return Interval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err, ctx=ctx)

    # helper for monotonically increasing functions
    def _monotonic_incr(op, lo_ep, hi_ep, err, ctxs):
        ctx, lo_ctx, hi_ctx = ctxs
        lo, lo_isfixed = Interval._epfn(op, lo_ep, ctx=lo_ctx)
        hi, hi_isfixed = Interval._epfn(op, hi_ep, ctx=hi_ctx)
        return Interval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err, ctx=ctx)

    # helper for monotonically decreasing functions
    def _monotonic_decr(op, lo_ep, hi_ep, err, ctxs):
        ctx, lo_ctx, hi_ctx = ctxs
        lo, lo_isfixed = Interval._epfn(op, hi_ep, ctx=lo_ctx)
        hi, hi_isfixed = Interval._epfn(op, lo_ep, ctx=hi_ctx)
        return Interval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err, ctx=ctx)

    # most operations

    def neg(self, ctx=None):
        """Negates this interval. The precision of the interval can be specified by `ctx`.
        """
        if self._invalid:
            return Interval(invalid=True)

        ctx, lo_ctx, hi_ctx = self._select_context(self, ctx=ctx)
        lo, lo_isfixed = type(self)._epfn(OP.neg, self._hi_endpoint(), ctx=lo_ctx)
        hi, hi_isfixed = type(self)._epfn(OP.neg, self._lo_endpoint(), ctx=hi_ctx)
        err = self._err
        return type(self)(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err, ctx=ctx)
    
    def add(self, other, ctx=None):
        """Adds this interval and another and returns the result.
        The precision of the interval can be specified by `ctx`."""
        if self._invalid or other._invalid:
            return Interval(invalid=True)

        ctx, lo_ctx, hi_ctx = self._select_context(self, other, ctx=ctx)
        lo, lo_isfixed = type(self)._eplinear(OP.add, self._lo_endpoint(), other._lo_endpoint(), lo_ctx)
        hi, hi_isfixed = type(self)._eplinear(OP.add, self._hi_endpoint(), other._hi_endpoint(), hi_ctx)
        err = self._err or other._err

        return type(self)(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err, ctx=ctx)

    def sub(self, other, ctx=None):
        """Subtracts this interval by another and returns the result.
        The precision of the interval can be specified by `ctx`."""
        if self._invalid or other._invalid:
            return Interval(invalid=True)

        ctx, lo_ctx, hi_ctx = self._select_context(self, other, ctx=ctx)
        lo, lo_isfixed = type(self)._eplinear(OP.sub, self._lo_endpoint(), other._hi_endpoint(), lo_ctx)
        hi, hi_isfixed = type(self)._eplinear(OP.sub, self._hi_endpoint(), other._lo_endpoint(), hi_ctx)
        err = self._err or other._err

        return type(self)(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err, ctx=ctx)

    def mul(self, other, ctx=None):
        """Multiplies this interval and another and returns the result.
        The precision of the interval can be specified by `ctx`."""
        if self._invalid or other._invalid:
            return Interval(invalid=True)

        xclass = self.classify()
        xlo = self._lo_endpoint()
        xhi = self._hi_endpoint()

        yclass = other.classify()
        ylo = other._lo_endpoint()
        yhi = other._hi_endpoint()

        ctxs = self._select_context(self, other, ctx=ctx)
        err = self._err or other._err

        if xclass == IntervalSign.POSITIVE:
            if yclass == IntervalSign.POSITIVE:
                return type(self)._multiply(xlo, ylo, xhi, yhi, xclass, yclass, err, ctxs)
            elif yclass == IntervalSign.NEGATIVE:
                return type(self)._multiply(xhi, ylo, xlo, yhi, xclass, yclass, err, ctxs)
            else:   # yclass == IntervalSign.ZERO
                return type(self)._multiply(xhi, ylo, xlo, yhi, xclass, yclass, err, ctxs)
        elif xclass == IntervalSign.NEGATIVE:
            if yclass == IntervalSign.POSITIVE:
                return type(self)._multiply(xlo, yhi, xhi, ylo, xclass, yclass, err, ctxs)
            elif yclass == IntervalSign.NEGATIVE:
                return type(self)._multiply(xhi, yhi, xlo, ylo, xclass, yclass, err, ctxs)
            else:   # yclass == IntervalSign.ZERO
                return type(self)._multiply(xlo, yhi, xlo, ylo, xclass, yclass, err, ctxs)
        else:   # xclass == IntervalSign.ZERO
            if yclass == IntervalSign.POSITIVE:
                return type(self)._multiply(xlo, yhi, xhi, yhi, xclass, yclass, err, ctxs)
            elif yclass == IntervalSign.NEGATIVE:
                return type(self)._multiply(xhi, ylo, xlo, ylo, xclass, yclass, err, ctxs)
            else:   # yclass == IntervalSign.ZERO
                i1 = type(self)._multiply(xhi, ylo, xlo, ylo, xclass, yclass, err, ctxs)
                i2 = type(self)._multiply(xlo, yhi, xhi, yhi, xclass, yclass, err, ctxs)
                return i1.union(i2)

    def div(self, other, ctx=None):
        """Divides this interval by another and returns the result.
        The precision of the interval can be specified by `ctx`."""
        if self._invalid or other._invalid:
            return Interval(invalid=True)

        xclass = self.classify()
        xlo = self._lo_endpoint()
        xhi = self._hi_endpoint()

        yclass = other.classify()
        ylo = other._lo_endpoint()
        yhi = other._hi_endpoint()

        ctxs = self._select_context(self, other, ctx=ctx)
        err = self.err or other.err or \
               (self.classify(strict=True) == IntervalSign.CONTAINS_ZERO and \
                other.classify(strict=True) == IntervalSign.CONTAINS_ZERO)

        if yclass == IntervalSign.CONTAINS_ZERO:
            # result is split, so we give up and always return [-inf, inf]
            isfixed = self._lo_isfixed and other._lo_isfixed
            return Interval(lo_isfixed=isfixed, hi_isfixed=isfixed)
        elif xclass == IntervalSign.POSITIVE:
            if yclass == IntervalSign.POSITIVE:
                return type(self)._divide(xlo, yhi, xhi, ylo, xclass, err, ctxs)
            else:   # yclass == IntervalSign.NEGATIVE
                return type(self)._divide(xhi, yhi, xlo, ylo, xclass, err, ctxs)
        elif xclass == IntervalSign.NEGATIVE:
            if yclass == IntervalSign.POSITIVE:
                return type(self)._divide(xlo, ylo, xhi, yhi, xclass, err, ctxs)
            else:   # yclass == IntervalSign.NEGATIVE
                return type(self)._divide(xhi, ylo, xlo, yhi, xclass, err, ctxs)
        else:   # xclass == IntervalSign.ZERO
            if yclass == IntervalSign.POSITIVE:
                return type(self)._divide(xlo, ylo, xhi, ylo, xclass, err, ctxs)
            else:   # yclass == IntervalSign.NEGATIVE
                return type(self)._divide(xhi, yhi, xlo, yhi, xclass, err, ctxs)
    
    def fabs(self, ctx=None):
        """Returns the absolute value of this interval.
        The precision of the interval can be specified by `ctx`.
        """
        if self._invalid:
            return Interval(invalid=True)

        ctx, lo_ctx, hi_ctx = self._select_context(self, ctx=ctx)
        cl = self.classify()
        if cl == IntervalSign.POSITIVE:
            return Interval(x=self)
        elif cl == IntervalSign.NEGATIVE:
            return self.neg()
        else:
            neg_lo, _ = type(self)._epfn(OP.neg, self._lo_endpoint(), ctx=hi_ctx)
            lo, lo_isfixed = ieee754.Float(0, ctx=lo_ctx), self._lo_isfixed and self._hi_isfixed
            hi, hi_isfixed = (neg_lo, self._lo_isfixed) if neg_lo > self._hi else (self._hi, self._hi_isfixed)
            return Interval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=self.err, ctx=ctx)

    def hypot(self, other, ctx=None):
        """Performs hypot(other1, other2) on this interval and another.
        The precision of the interval can be specified by `ctx`.
        """
        if self._invalid or other._invalid:
            return Interval(invalid=True)

        posx = self.fabs(self._ctx)
        posy = other.fabs(other._ctx)
        err = self._err and other._err

        ctx, lo_ctx, hi_ctx = self._select_context(self, ctx=ctx)
        lo, lo_isfixed = type(self)._eplinear(OP.hypot, posx._lo_endpoint(), posy._lo_endpoint(), lo_ctx)
        hi, hi_isfixed = type(self)._eplinear(OP.hypot, posx._hi_endpoint(), posy._hi_endpoint(), hi_ctx)
        return Interval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err, ctx=ctx)
    
    def fma(self, other1, other2, ctx=None):
        """Performs fused-multiply add on this interval and two others.
        The precision of the interval can be specified by `ctx`.
        Note: naively implemented by multiplication and division
        """
        return self.mul(other1, ctx).add(other2, ctx)

    def sqrt(self, ctx=None):
        """Returns the square root of this interval.
        The precision of the interval can be specified by `ctx`.
        """
        if self._invalid:
            return Interval(invalid=True)

        clamped = self.clamp(ieee754.Float(0.0), digital.Digital(negative=False, isinf=True), err=True)
        if clamped.invalid:
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        return type(self)._monotonic_incr(OP.sqrt, clamped._lo_endpoint(), clamped._hi_endpoint(), clamped.err, ctxs)

    def cbrt(self, ctx=None):
        """Returns the cube root of this interval.
        The precision of the interval can be specified by `ctx`.
        """
        if self._invalid:
            return Interval(invalid=True)
        
        ctxs = self._select_context(self, ctx=ctx)
        return type(self)._monotonic_incr(OP.cbrt, self._lo_endpoint(), self._hi_endpoint(), self._err, ctxs)

#
#   Testing
#

import random

def random_float(min=None, max=None, ctx=None):
    if ctx is None:
        ctx = ieee754.ieee_ctx(11, 64)
    inf_ord = ieee754.digital_to_bits(x=ieee754.Float('inf', ctx=ctx), ctx=ctx)

    if min is not None:
        min_ord = ieee754.digital_to_bits(min, ctx=ctx)
    else:
        min_ord = 1 - inf_ord
    
    if max is not None:
        max_ord = ieee754.digital_to_bits(max, ctx=ctx)
    else:
        max_ord = inf_ord - 1

    lo_ord = random.randint(min_ord, max_ord)
    return ieee754.bits_to_digital(lo_ord, ctx=ctx)

def random_interval(min=None, max=None, ctx=None):
    if ctx is None:
        ctx = ieee754.ieee_ctx(11, 64)
    inf_ord = ieee754.digital_to_bits(x=ieee754.Float('inf', ctx=ctx), ctx=ctx)

    if min is not None:
        min_ord = ieee754.digital_to_bits(min, ctx=ctx)
    else:
        min_ord = 1 - inf_ord
    
    if max is not None:
        max_ord = ieee754.digital_to_bits(max, ctx=ctx)
    else:
        max_ord = inf_ord - 1

    lo_ord = random.randint(min_ord, max_ord)
    lo = ieee754.bits_to_digital(lo_ord, ctx=ctx)

    hi_ord = random.randint(min_ord, max_ord)
    hi = ieee754.bits_to_digital(hi_ord, ctx=ctx)

    # switch if needed
    if lo > hi:
        lo, hi = hi, lo

    return Interval(lo=lo, hi=hi)

def test_unary_fn(name, fl_fn, ival_fn, num_tests, ctx):
    bad = []
    for _ in range(num_tests):
        xf = random_float(ctx=ctx)
        fl = fl_fn(xf, ctx=ctx)

        x = Interval(xf, ctx=ctx)
        ival = ival_fn(x, ctx=ctx)

        if fl.isnan and not ival.invalid:
            bad.append((xf, fl, ival, ival.err, ival.invalid))
        else:
            if ival.err or fl < ival.lo or fl > ival.hi:
                bad.append((xf, fl, ival, ival.err, ival.invalid))
    if len(bad) > 0:
        print('[FAILED] {} {}/{}'.format(name, len(bad), num_tests))
        for xf, fl, ival, err, invalid in bad:
            print(' x={} fl={}, ival={}, err={}, invalid={}'.format(
                str(xf), str(fl), str(ival), err, invalid))
    else:
        print('[PASSED] {}'.format(name))

def test_binary_fn(name, fl_fn, ival_fn, num_tests, ctx):
    bad = []
    for _ in range(num_tests):
        xf = random_float(ctx=ctx)
        yf = random_float(ctx=ctx)
        fl = fl_fn(xf, yf, ctx=ctx)

        x = Interval(xf, ctx=ctx)
        y = Interval(yf, ctx=ctx)
        ival = ival_fn(x, y, ctx=ctx)

        if fl.isnan and not ival.invalid:
            bad.append((xf, yf, fl, ival, ival.err, ival.invalid))
        else:
            if ival.err or fl < ival.lo or fl > ival.hi:
                bad.append((xf, yf, fl, ival, ival.err, ival.invalid))
    if len(bad) > 0:
        print('[FAILED] {} {}/{}'.format(name, len(bad), num_tests))
        for xf, fl, yf, ival, err, invalid in bad:
            print(' x={} y={} fl={}, ival={}, err={}, invalid={}'.format(
                str(xf), str(yf), str(fl), str(ival), err, invalid))
    else:
        print('[PASSED] {}'.format(name))

def test_interval():
    """Runs unit tests for the Interval type"""
    num_tests = 10_000
    ctx = ieee754.ieee_ctx(11, 64)
    random.seed()

    ops = [
        ("neg", ieee754.Float.neg, Interval.neg, 1),
        ("fabs", ieee754.Float.fabs, Interval.fabs, 1),
        ("add", ieee754.Float.add, Interval.add, 2),
        ("sub", ieee754.Float.sub, Interval.sub, 2),
        ("mul", ieee754.Float.mul, Interval.mul, 2),
        ("div", ieee754.Float.div, Interval.div, 2),
        ("sqrt", ieee754.Float.sqrt, Interval.sqrt, 1),
        ("cbrt", ieee754.Float.cbrt, Interval.cbrt, 1)
    ]

    # ops = [("add", ieee754.Float.add, Interval.add, 2)]

    for name, fl_fn, ival_fn, argc in ops:
        if argc == 1:
            test_unary_fn(name, fl_fn, ival_fn, num_tests, ctx)
        elif argc == 2:
            test_binary_fn(name, fl_fn, ival_fn, num_tests, ctx)
        else:
            raise ValueError('invalid argc for testing: argc={}'.format(argc))
