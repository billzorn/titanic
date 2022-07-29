"""Interval arithmetic based on Rival
Original implementation: https://github.com/herbie-fp/rival
"""

from titanfp.titanic import gmpmath
from . import interpreter
from . import ieee754

from ..titanic import digital
from ..titanic.ops import OP, RM

class Interval(object):
    """An interval as originally implemented in the Rival interval arithmetic library.
    Original implementation at https://github.com/herbie-fp/rival written by
    Pavel Panchekha and Oliver Flatt. The interval uses MPFR floating-point value bounds,
    immovability flags to signal a fixed endpoint due to overflow, and error flags to propogate
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
    _lo_ctx : ieee754.IEEECtx = ieee754.ieee_ctx(11, 64, rm=RM.RTN)
    _hi_ctx : ieee754.IEEECtx = ieee754.ieee_ctx(11, 64, rm=RM.RTP)

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

        # _ctx (_lo_ctx and _hi_ctx)
        if ctx is not None:
            self._ctx = ctx
            self._lo_ctx = ieee754.ieee_ctx(es=ctx.es, nbits=ctx.nbits, rm=RM.RTN)
            self._hi_ctx = ieee754.ieee_ctx(es=ctx.es, nbits=ctx.nbits, rm=RM.RTP)
        elif x is not None:
            self._ctx = x._ctx
            self._lo_ctx = x._lo_ctx
            self._hi_ctx = x._hi_ctx
        else:
            self._ctx = type(self)._ctx
            self._lo_ctx = type(self)._lo_ctx
            self._hi_ctx = type(self)._hi_ctx

        # _lo and _hi
        if x is not None:
            if lo is not None or hi is not None:
                raise ValueError('cannot specify both x={} and [lo={}, hi={}]'.format(repr(x), repr(lo), repr(hi)))
            if isinstance(x, type(self)):
                self._lo = ieee754.Float(x._lo, ctx=self._lo_ctx)
                self._hi = ieee754.Float(x._hi, ctx=self._hi_ctx)
            else:
                self._lo = ieee754.Float(x, ctx=self._lo_ctx)
                self._hi = ieee754.Float(x, ctx=self._hi_ctx)
        elif lo is not None:
            if hi is None:
                raise ValueError('must specify both lo={} and hi={} together'.format(repr(lo), repr(hi)))
            if x is not None:
                raise ValueError('cannot specify both x={} and [lo={}, hi={}]'.format(repr(x), repr(lo), repr(hi)))
            self._lo = ieee754.Float(x=lo, ctx=self._lo_ctx)
            self._hi = ieee754.Float(x=hi, ctx=self._hi_ctx)
            if self._lo > self._hi:
                raise ValueError()                
        else:
            self._lo = type(self)._lo
            self._hi = type(self)._hi

        # _lo_isfixed
        if lo_isfixed:
            self._lo_isfixed = lo_isfixed
        elif x is not None:
            self._lo_isfixed = x._lo_isfixed
        else:
            self._lo_isfixed = type(self)._lo_isfixed

        # _hi_isfixed
        if hi_isfixed:
            self._hi_isfixed = hi_isfixed
        elif x is not None:
            self._hi_isfixed = x._hi_isfixed
        else:
            self._hi_isfixed = type(self)._hi_isfixed

        # _err
        if err:
            self._err = err
        elif x is not None:
            self._err = x._err
        else:
            self._err = type(self)._err

        # _invalid
        if invalid:
            self._invalid = invalid
        elif x is not None:
            self._invalid = x._invalid
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

    # some predicates

    def is_point(self) -> bool:
        """"Is the interval a "singleton" interval, e.g. [a, a]?"""
        return self._lo == self._hi

    def contains(self, x) -> bool:
        """Does the interval contain `x`?"""
        return self._lo <= x and x <= self._hi

    # utility funtions

    def _lo_endpoint(self):
        return (self._lo, self._lo_endpoint)

    def _hi_endpoint(self):
        return (self._hi, self._hi_endpoint)

    def _select_context(self, *args, ctx=None):
        if ctx is None:
            es = max((ival._ctx.es for ival in args))
            p = max((ival._ctx.p for ival in args))
            es = max(es, self._ctx.es)
            p = max(p, self._ctx.p)
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
        isfixed = args_fixed and not result.inexact
        rounded = ieee754.Float(x=result, ctx=ctx)
        return rounded, isfixed

    # endpoint computation for `add`, `sub`, `hypot`
    def _eplinear(op, lo_ep, hi_ep, ctx):
        lo, lo_isfixed = lo_ep
        hi, hi_isfixed = hi_ep
        result = gmpmath.compute(op, lo, hi, prec=ctx.p)
        isfixed = (lo_isfixed and hi_isfixed and not result.inexact) or \
                           (lo_isfixed and lo.isinf) or \
                           (hi_isfixed and hi.isinf)
        rounded = ieee754.Float(x=result, ctx=ctx)
        return rounded, isfixed

    # endpoint computation for `mul`
    # def _epmul(op, lo_ep, hi_ep, ctx):
    #     lo, lo_isfixed = lo_ep
    #     hi, hi_isfixed = hi_ep
    #     result = gmpmath.compute(op, lo, hi, prec=ctx.p)
    #     endpoint_isfixed = (lo_isfixed and hi_isfixed and not result.inexact) or \
    #                        (lo_isfixed and lo.isinf) or \
    #                        (hi_isfixed and hi.isinf)
    #     rounded = ieee754.Float(x=result, ctx=ctx)
    #     return rounded, endpoint_isfixed

    # most operations

    def neg(self, ctx=None):
        """Negates this interval. The precision of the interval can be specified by `ctx`.
        Optionally specify `rounded` to round at the current precision
        """
        ctx, lo_ctx, hi_ctx = self._select_context(self, ctx=ctx)
        lo, lo_isfixed = type(self)._epfn(OP.neg, self._hi_endpoint(), ctx=lo_ctx)
        hi, hi_isfixed = type(self)._epfn(OP.neg, self._lo_endpoint(), ctx=hi_ctx)
        err = self._err
        invalid = self._invalid
        return type(self)(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err, invalid=invalid, ctx=ctx)
    
    def add(self, other, ctx=None):
        """Adds two intervals together and returns the result.
        The precision of the interval can be specified by `ctx`."""
        ctx, lo_ctx, hi_ctx = self._select_context(self, other, ctx=ctx)
        lo, lo_isfixed = type(self)._eplinear(OP.add, self._lo_endpoint(), other._lo_endpoint(), lo_ctx)
        hi, hi_isfixed = type(self)._eplinear(OP.add, self._hi_endpoint(), other._hi_endpoint(), hi_ctx)
        err = self._err or other._err
        invalid = self._invalid or other._invalid
        return type(self)(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err, invalid=invalid, ctx=ctx)

    def sub(self, other, ctx=None):
        """Subtracts two intervals together and returns the result.
        The precision of the interval can be specified by `ctx`."""
        ctx, lo_ctx, hi_ctx = self._select_context(self, other, ctx=ctx)
        lo, lo_isfixed = type(self)._eplinear(OP.sub, self._lo_endpoint(), other._hi_endpoint(), lo_ctx)
        hi, hi_isfixed = type(self)._eplinear(OP.sub, self._hi_endpoint(), other._lo_endpoint(), hi_ctx)
        err = self._err or other._err
        invalid = self._invalid or other._invalid
        return type(self)(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err, invalid=invalid, ctx=ctx)
