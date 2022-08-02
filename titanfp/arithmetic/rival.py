"""Interval arithmetic based on Rival
Original implementation: https://github.com/herbie-fp/rival
"""

from enum import IntEnum, unique
import gmpy2 as gmp
import math

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

@unique
class IntervalOrder(IntEnum):
    """Classification of an `Interval` by comparison against a value.
    See `Interval.compare()` for details."""
    STRICTLY_LESS     = -2,
    LESS              = -1,
    CONTAINS          = 0
    GREATER           = 1,
    STRICTLY_GREATER  = 2,

#
#   Unsupported floating-point operations
#

def rint(x: ieee754.Float, ctx):
    """Basically just `gmpmath.compute()` specialized for `rint` until
    Titanic officially supports `rint`."""
    ctx = x._select_context(x, ctx=ctx)
    input = gmpmath.digital_to_mpfr(x)
    # gmpy2 really doesn't like it when you pass nan as an argument
    if gmp.is_nan(input):
        return gmpmath.mpfr_to_digital(input)
    with gmp.context(
            # one extra bit, so that we can round from RTZ to RNE
            precision=ctx.p + 1,
            emin=gmp.get_emin_min(),
            emax=gmp.get_emax_max(),
            subnormalize=False,
            # unlike `gmpmath.compute()`, these are disabled
            trap_underflow=False,
            trap_overflow=False,
            # inexact and invalid operations should not be a problem
            trap_inexact=False,
            trap_invalid=False,
            trap_erange=False,
            trap_divzero=False,
            # We'd really like to know about this as well, but it causes i.e.
            #   mul(-25, inf) -> raise TypeError("mul() requires 'mpfr','mpfr' arguments")
            # I don't know if that behavior is more hilarious or annoying.
            # This context property was removed in gmpy2 2.1
            #trap_expbound=False,
            # use RTZ for easy multiple rounding later
            round=gmp.RoundToZero,
    ) as gmpctx:
        result = gmp.rint(input)
        if gmpctx.underflow:
            result = gmp.mpfr(0)
        elif gmpctx.overflow:
            result = gmp.inf(gmp.sign(result))

    return x._round_to_context(gmpmath.mpfr_to_digital(result, ignore_rc=True), ctx=ctx, strict=True)


def is_integer(x):
    """Takes a Digital type `x` and returns true if `x` is an integer."""
    if not isinstance(x, digital.Digital):
        raise ValueError('expected a Digital type: {}'.format(x))
    elif x.isinf or x.isnan:
        return False
    elif x.is_zero() or x.exp >= 0:
        return True
    elif x.exp > x.c.bit_length():
        return False
    else:
        return (x.c & ((2 ** -x.exp) - 1)) == 0

def is_even_integer(x):
    """Takes a Digital type `x` and returns true if `x` is an even integer."""
    if not isinstance(x, digital.Digital):
        raise ValueError('expected a Digital type: {}'.format(x))
    elif not is_integer(x):
        return False
    elif x.is_zero() or x.exp > 0:
        return True
    elif -x.exp > x.c.bit_length():
        return True
    else:
        return (abs(x.c) & (2 ** -x.exp)) == 0

def is_odd_integer(x):
    """Takes a Digital type `x` and returns true if `x` is an odd integer."""
    if not isinstance(x, digital.Digital):
        raise ValueError('expected a Digital type: {}'.format(x))
    elif not is_integer(x):
        return False
    elif x.is_zero() or x.exp > 0:
        return False
    elif -x.exp > x.c.bit_length():
        return False
    else:
        return (abs(x.c) & (2 ** -x.exp)) != 0

#
#   Interval endpoint computation
#

# simple endpoint computation that propagates immovability
def _epfn(op, *eps, ctx):
    args = []
    args_fixed = False
    for ival, isfixed in eps:
        args.append(ival)
        args_fixed = args_fixed and isfixed

    result = gmpmath.compute(op, *args, prec=ctx.p, trap_underflow=False, trap_overflow=False)
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
def _epmul(a_ep, b_ep, aclass, bclass, ctx):
    a, a_isfixed, = a_ep
    b, b_isfixed, = b_ep
    result = gmpmath.compute(OP.mul, a, b, prec=ctx.p)
    rounded = ieee754.Float._round_to_context(result, ctx=ctx)
    isfixed = (a_isfixed and b_isfixed and not rounded.inexact) or \
                (a_isfixed and a.is_zero() and not a.isinf) or \
                (a_isfixed and a.isinf and bclass != IntervalSign.CONTAINS_ZERO) or \
                (b_isfixed and b.is_zero() and not b.isinf) or \
                (b_isfixed and b.isinf and aclass != IntervalSign.CONTAINS_ZERO)
    return rounded, isfixed

# endpoint computation for `div`
def _epdiv(a_ep, b_ep, aclass, ctx):
    a, a_isfixed, = a_ep
    b, b_isfixed, = b_ep
    result = gmpmath.compute(OP.div, a, b, prec=ctx.p)
    rounded = ieee754.Float._round_to_context(result, ctx=ctx)
    isfixed = (a_isfixed and b_isfixed and not rounded.inexact) or \
                (a_isfixed and a.is_zero() and not a.isinf) or \
                (a_isfixed and a.isinf) or \
                (b_isfixed and b.is_zero() and not b.isinf) or \
                (b_isfixed and b.isinf and aclass != IntervalSign.CONTAINS_ZERO)
    return rounded, isfixed

# endpoint computation for `rint` (really just `epfn`)
def _eprint(ep, ctx):
    arg, arg_isfixed = ep
    rounded = rint(arg, ctx)
    isfixed = arg_isfixed and not rounded.inexact
    return rounded, isfixed

# endpoint computation or `pow`
def _eppow(a_ep, b_ep, aclass, bclass, ctx):
    a, a_isfixed, = a_ep
    b, b_isfixed, = b_ep
    result = gmpmath.compute(OP.pow, a, b, prec=ctx.p, trap_underflow=False, trap_overflow=False)
    rounded = ieee754.Float._round_to_context(result, ctx=ctx)
    isfixed = (a_isfixed and b_isfixed and not rounded.inexact) or \
                (a_isfixed and a == digital.Digital(c=1, exp=0)) or \
                (a_isfixed and a == digital.Digital(c=0, exp=0) and bclass != IntervalSign.CONTAINS_ZERO) or \
                (a_isfixed and a.isinf and bclass != IntervalSign.CONTAINS_ZERO) or \
                (b_isfixed and b == digital.Digital(c=0, exp=0)) or \
                (b_isfixed and b.isinf and aclass != IntervalSign.CONTAINS_ZERO)
    return rounded, isfixed

#
#   Specialized cases of interval functions
#

def _multiply(a, b, c, d, xclass, yclass, err, ctxs):
    ctx, lo_ctx, hi_ctx = ctxs
    lo, lo_isfixed = _epmul(a, b, xclass, yclass, lo_ctx)
    hi, hi_isfixed = _epmul(c, d, xclass, yclass, hi_ctx)
    return Interval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err, ctx=ctx)

def _divide(a, b, c, d, xclass, err, ctxs):
    ctx, lo_ctx, hi_ctx = ctxs
    lo, lo_isfixed = _epdiv(a, b, xclass, lo_ctx)
    hi, hi_isfixed = _epdiv(c, d, xclass, hi_ctx)
    return Interval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err, ctx=ctx)

def _power(a, b, c, d, x, y, xclass, yclass, err, ctxs):
    ctx, lo_ctx, hi_ctx = ctxs
    lo, lo_isfixed = _eppow(a, b, xclass, yclass, lo_ctx)
    hi, hi_isfixed = _eppow(c, d, xclass, yclass, hi_ctx)
    result = Interval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err, ctx=ctx)
    # HACK: `is_zero` checks for x == 0 and x == +/-INF simultaneously
    if result._lo.is_zero() or result._hi.is_zero():
        movability = y.mul(x.log()).exp()
        return Interval(lo=result._lo,
                        hi=result._hi,
                        lo_isfixed=movability._lo_isfixed,
                        hi_isfixed=movability._hi_isfixed,
                        err=result._err,
                        ctx=result._ctx)
    else:
        return result

# assumes `x` is positive
def _power_pos(x, y, ctxs):
    if x._invalid or y._invalid:
        return Interval(invalid=True)

    xclass = x.compare(ieee754.Float(1.0))
    xlo = x._lo_endpoint()
    xhi = x._hi_endpoint()

    yclass = y.classify()
    ylo = y._lo_endpoint()
    yhi = y._hi_endpoint()

    err = x._err or y._err
    if xclass == IntervalOrder.GREATER:
        if yclass == IntervalOrder.GREATER:
            return _power(xlo, ylo, xhi, yhi, x, y, xclass, yclass, err, ctxs)
        elif yclass == IntervalOrder.LESS:
            return _power(xhi, ylo, xlo, yhi, x, y, xclass, yclass, err, ctxs)
        else:   # yclass == IntervalOrder.CONTAINS
            return _power(xhi, ylo, xhi, yhi, x, y, xclass, yclass, err, ctxs)
    elif xclass == IntervalOrder.LESS:
        if yclass == IntervalOrder.GREATER:
            return _power(xlo, yhi, xhi, ylo, x, y, xclass, yclass, err, ctxs)
        elif yclass == IntervalOrder.LESS:
            return _power(xhi, yhi, xlo, ylo, x, y, xclass, yclass, err, ctxs)
        else:   # yclass == IntervalOrder.CONTAINS
            return _power(xlo, yhi, xlo, ylo, x, y, xclass, yclass, err, ctxs)
    else:   # xclass == IntervalOrder.CONTAINS
        if yclass == IntervalOrder.GREATER:
            return _power(xlo, yhi, xhi, yhi, x, y, xclass, yclass, err, ctxs)
        elif yclass == IntervalOrder.LESS:
            return _power(xhi, ylo, xlo, ylo, x, y, xclass, yclass, err, ctxs)
        else:   # yclass == IntervalOrder.CONTAINS
            i1 = _power(xlo, yhi, xhi, yhi, x, y, xclass, yclass, err, ctxs)
            i2 = _power(xhi, ylo, xlo, ylo, x, y, xclass, yclass, err, ctxs)
            return i1.union(i2)

# assumes `x` is negative
def _power_neg(x, y, ctxs):
    if x._invalid or y._invalid or y._lo < y._hi:
        return Interval(invalid=True)

    ctx, lo_ctx, hi_ctx = ctxs
    err = x.err or y.err
    pos_x = x.fabs()
    a = y._lo.ceil(ctx)
    b = y._hi.floor(ctx)
    if a > b:
        if x._hi == ieee754.Float(0.0):
            return Interval(lo=0.0, hi=0.0, lo_isfixed=False, hi_isfixed=False, err=True)
        else:
            return Interval(invalid=True)
    elif a == b:
        a_isfixed = y._lo_isfixed and y._hi_isfixed
        p = Interval(lo=a, hi=a, lo_isfixed=a_isfixed, hi_isfixed=a_isfixed, err=err)
        if is_odd_integer(a):
            return _power_pos(pos_x, p, ctxs).neg()
        else:
            return _power_pos(pos_x, p, ctxs)
    else:
        # movability flags not implemented
        # original implementation just sets them to False
        one = ieee754.Float(1.0)
        odd_lo = a if is_odd_integer(a) else a.add(one, lo_ctx)
        odd_hi = b if is_odd_integer(b) else b.sub(one, hi_ctx)
        even_lo = a.add(one, lo_ctx) if is_odd_integer(a) else a
        even_hi = b.sub(one, hi_ctx) if is_odd_integer(b) else b
        odds = Interval(lo=odd_lo, hi=odd_hi, err=err)
        evens = Interval(lo=even_lo, hi=even_hi, err=err)
        i1 = _power_pos(pos_x, evens)
        i2 = _power_pos(pos_x, odds).neg()
        return i1.union(i2)

def _monotonic_incr(op, lo_ep, hi_ep, err, ctxs):
    ctx, lo_ctx, hi_ctx = ctxs
    lo, lo_isfixed = _epfn(op, lo_ep, ctx=lo_ctx)
    hi, hi_isfixed = _epfn(op, hi_ep, ctx=hi_ctx)
    return Interval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err, ctx=ctx)

def _monotonic_decr(op, lo_ep, hi_ep, err, ctxs):
    ctx, lo_ctx, hi_ctx = ctxs
    lo, lo_isfixed = _epfn(op, hi_ep, ctx=lo_ctx)
    hi, hi_isfixed = _epfn(op, lo_ep, ctx=hi_ctx)
    return Interval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err, ctx=ctx)

def _overflow(lo, hi, x, y):
    lo_isfixed = y._lo_isfixed or (x._hi <= lo) or (x._lo <= lo and x._lo_isfixed)
    hi_isfixed = y._hi_isfixed or (x._lo >= hi) or (x._hi >= hi and x._hi_isfixed)
    return Interval(lo=y._lo, hi=y._hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=x._err, ctx=y._ctx)

#
#   Digital constants
#

with gmp.local_context(gmp.context(), trap_inexact=False):
    EXP_OVERFLOW =  gmpmath.mpfr_to_digital(gmp.log(gmp.next_below(gmp.inf())) + gmp.mpfr(1))
    EXP2_OVERFLOW =  gmpmath.mpfr_to_digital(gmp.log2(gmp.next_below(gmp.inf())) + gmp.mpfr(1))

#
#   Interval type
#

class Interval(object):
    """An interval based on the Rival interval arithmetic library.
    Original implementation at https://github.com/herbie-fp/rival written by Pavel Panchekha and Oliver Flatt.
    The interval uses MPFR floating-point value bounds, immovability flags to signal a fixed endpoint
    due to overflow, and error flags to propagate partial or complete domain errors.
    """

    # represents the real number interval [_lo, _hi]
    # (using ieee754.Float since this makes the most sense)
    _lo: ieee754.Float = ieee754.Float(digital.Digital(negative=True, isinf=True), ctx=ieee754.ieee_ctx(11, 64))
    _hi: ieee754.Float = ieee754.Float(digital.Digital(negative=False, isinf=True), ctx=ieee754.ieee_ctx(11, 64))

    # immovability flags
    # (from the original Rival implementation:
    #   "Intervals may shrink (though they cannot grow) when computed at a higher precision.
    #   However, in some cases it is known this will not occur, largely due to overflow.
    #   In those cases, the interval is marked fixed, or immovable."
    # )
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
        If `strict` is `True`, this interval can be `STRICTLY_NEGATIVE`, `CONTAINS_ZERO`
        or `STRICTLY_POSITIVE` where intervals with 0 as an endpoint are classified
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

    def compare(self, val, strict=False) -> IntervalOrder:
        """Classifies this interval by comparing against a value.
        By default, this interval can be `LESS`, `CONTAINS`, or `GREATER`
        where intervals with `val` as an endpoint are either `LESS` and `GREATER`.
        If `strict` is `True`, this interval can be `STRICTLY_LESS`, `CONTAINS`
        or `STRICTLY_GREATER` where intervals with `val` as an endpoint are classified
        as `CONTAINS`.
        """
        if strict:
            if self._hi < val:
                return IntervalOrder.STRICTLY_LESS
            elif self._lo > val:
                return IntervalOrder.STRICTLY_GREATER
            else:
                return IntervalOrder.CONTAINS
        else:
            if self._hi <= val:
                return IntervalOrder.LESS
            elif self._lo >= val:
                return IntervalOrder.GREATER
            else:
                return IntervalOrder.CONTAINS

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

    def split(self, val):
        """Takes a finite value `val` between `self.lo` and `self.hi` and
        returns two intervals: `[self.lo, val]` and `[val, self.hi]`."""
        if not isinstance(val):
            raise ValueError('expected a Digital type {}'.format(val))
        if val.isnan or val.isinf:
            raise ValueError('cannot split on a finite value {}'.format(val))
        if val < self._lo or val > self._hi:
            raise ValueError('split value must be between the endpoints {}'.format(val))
        if self.isnan:
            return Interval(invalid=True), Interval(invalid=True)
        return Interval(x=self, hi=val), Interval(x=self, lo=val)

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

    # most operations

    def neg(self, ctx=None):
        """Negates this interval. The precision of the interval can be specified by `ctx`.
        """
        if self._invalid:
            return Interval(invalid=True)

        ctx, lo_ctx, hi_ctx = self._select_context(self, ctx=ctx)
        lo, lo_isfixed = _epfn(OP.neg, self._hi_endpoint(), ctx=lo_ctx)
        hi, hi_isfixed = _epfn(OP.neg, self._lo_endpoint(), ctx=hi_ctx)
        err = self._err
        return type(self)(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err, ctx=ctx)
    
    def add(self, other, ctx=None):
        """Adds this interval and another and returns the result.
        The precision of the interval can be specified by `ctx`."""
        if self._invalid or other._invalid:
            return Interval(invalid=True)

        ctx, lo_ctx, hi_ctx = self._select_context(self, other, ctx=ctx)
        lo, lo_isfixed = _eplinear(OP.add, self._lo_endpoint(), other._lo_endpoint(), lo_ctx)
        hi, hi_isfixed = _eplinear(OP.add, self._hi_endpoint(), other._hi_endpoint(), hi_ctx)
        err = self._err or other._err

        return type(self)(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err, ctx=ctx)

    def sub(self, other, ctx=None):
        """Subtracts this interval by another and returns the result.
        The precision of the interval can be specified by `ctx`."""
        if self._invalid or other._invalid:
            return Interval(invalid=True)

        ctx, lo_ctx, hi_ctx = self._select_context(self, other, ctx=ctx)
        lo, lo_isfixed = _eplinear(OP.sub, self._lo_endpoint(), other._hi_endpoint(), lo_ctx)
        hi, hi_isfixed = _eplinear(OP.sub, self._hi_endpoint(), other._lo_endpoint(), hi_ctx)
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
                return _multiply(xlo, ylo, xhi, yhi, xclass, yclass, err, ctxs)
            elif yclass == IntervalSign.NEGATIVE:
                return _multiply(xhi, ylo, xlo, yhi, xclass, yclass, err, ctxs)
            else:   # yclass == IntervalSign.ZERO
                return _multiply(xhi, ylo, xlo, yhi, xclass, yclass, err, ctxs)
        elif xclass == IntervalSign.NEGATIVE:
            if yclass == IntervalSign.POSITIVE:
                return _multiply(xlo, yhi, xhi, ylo, xclass, yclass, err, ctxs)
            elif yclass == IntervalSign.NEGATIVE:
                return _multiply(xhi, yhi, xlo, ylo, xclass, yclass, err, ctxs)
            else:   # yclass == IntervalSign.ZERO
                return _multiply(xlo, yhi, xlo, ylo, xclass, yclass, err, ctxs)
        else:   # xclass == IntervalSign.ZERO
            if yclass == IntervalSign.POSITIVE:
                return _multiply(xlo, yhi, xhi, yhi, xclass, yclass, err, ctxs)
            elif yclass == IntervalSign.NEGATIVE:
                return _multiply(xhi, ylo, xlo, ylo, xclass, yclass, err, ctxs)
            else:   # yclass == IntervalSign.ZERO
                i1 = _multiply(xhi, ylo, xlo, ylo, xclass, yclass, err, ctxs)
                i2 = _multiply(xlo, yhi, xhi, yhi, xclass, yclass, err, ctxs)
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
                return _divide(xlo, yhi, xhi, ylo, xclass, err, ctxs)
            else:   # yclass == IntervalSign.NEGATIVE
                return _divide(xhi, yhi, xlo, ylo, xclass, err, ctxs)
        elif xclass == IntervalSign.NEGATIVE:
            if yclass == IntervalSign.POSITIVE:
                return _divide(xlo, ylo, xhi, yhi, xclass, err, ctxs)
            else:   # yclass == IntervalSign.NEGATIVE
                return _divide(xhi, ylo, xlo, yhi, xclass, err, ctxs)
        else:   # xclass == IntervalSign.ZERO
            if yclass == IntervalSign.POSITIVE:
                return _divide(xlo, ylo, xhi, ylo, xclass, err, ctxs)
            else:   # yclass == IntervalSign.NEGATIVE
                return _divide(xhi, yhi, xlo, yhi, xclass, err, ctxs)
    
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
            neg_lo, _ = _epfn(OP.neg, self._lo_endpoint(), ctx=hi_ctx)
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
        lo, lo_isfixed = _eplinear(OP.hypot, posx._lo_endpoint(), posy._lo_endpoint(), lo_ctx)
        hi, hi_isfixed = _eplinear(OP.hypot, posx._hi_endpoint(), posy._hi_endpoint(), hi_ctx)
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
        return _monotonic_incr(OP.sqrt, clamped._lo_endpoint(), clamped._hi_endpoint(), clamped.err, ctxs)

    def cbrt(self, ctx=None):
        """Returns the cube root of this interval.
        The precision of the interval can be specified by `ctx`.
        """
        if self._invalid:
            return Interval(invalid=True)
        
        ctxs = self._select_context(self, ctx=ctx)
        return _monotonic_incr(OP.cbrt, self._lo_endpoint(), self._hi_endpoint(), self._err, ctxs)

    def rint(self, ctx=None):
        """Rounds this interval to a nearby integer.
        Normally, `rint` respects rounding mode, but interval arithmetic will round conservatively.
        The precision of the interval can be specified by `ctx`."""
        if self._invalid:
            return Interval(invalid=True)

        ctx, lo_ctx, hi_ctx = self._select_context(self, ctx=ctx)
        lo, lo_isfixed = _eprint(self._lo_endpoint(), ctx=lo_ctx)
        hi, hi_isfixed = _eprint(self._hi_endpoint(), ctx=hi_ctx)
        return Interval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=self._err, ctx=ctx)

    def round(self, ctx=None):
        """Rounds the endpoints of interval to the nearest integer.
        The precision of the interval can be specified by `ctx`."""
        if self._invalid:
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        return _monotonic_incr(OP.round, self._lo_endpoint(), self._hi_endpoint(), self._err, ctxs)

    def ceil(self, ctx=None):
        """Rounds each endpoint of the interval to the smallest integer not
        less than the endpoint. The precision of the interval can be specified by `ctx`."""
        if self._invalid:
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        return _monotonic_incr(OP.ceil, self._lo_endpoint(), self._hi_endpoint(), self._err, ctxs)

    def floor(self, ctx=None):
        """Rounds each endpoint of the interval to the largest integer not
        more than the endpoint. The precision of the interval can be specified by `ctx`."""
        if self._invalid:
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        return _monotonic_incr(OP.floor, self._lo_endpoint(), self._hi_endpoint(), self._err, ctxs)

    def trunc(self, ctx=None):
        """Rounds each endpoint of the interval to the nearest integer not
        larger in magnitude than the endpoint. The precision of the interval
        can be specified by `ctx`."""
        if self._invalid:
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        return _monotonic_incr(OP.trunc, self._lo_endpoint(), self._hi_endpoint(), self._err, ctxs)

    def exp(self, ctx=None):
        """Computes e^x on this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid:
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        result = _monotonic_incr(OP.exp, self._lo_endpoint(), self._hi_endpoint(), self._err, ctxs)
        return _overflow(digital.Digital(x=EXP_OVERFLOW, negative=True), EXP_OVERFLOW, self, result)

    def exp_(self, ctx=None):
        """Alternative name for `Interval.exp()`"""
        return self.exp(ctx=ctx)
    
    def expm1(self, ctx=None):
        """Computes e^x - 1 on this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid:
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        result = _monotonic_incr(OP.expm1, self._lo_endpoint(), self._hi_endpoint(), self._err, ctxs)
        return _overflow(digital.Digital(x=EXP_OVERFLOW, negative=True), EXP_OVERFLOW, self, result)

    def exp2(self, ctx=None):
        """Computes 2^x on this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid:
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        result = _monotonic_incr(OP.exp2, self._lo_endpoint(), self._hi_endpoint(), self._err, ctxs)
        return _overflow(digital.Digital(x=EXP2_OVERFLOW, negative=True), EXP2_OVERFLOW, self, result)
    
    def log(self, ctx=None):
        """Computes log(x) on this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid:
            return Interval(invalid=True)

        clamped = self.clamp(ieee754.Float(0.0), digital.Digital(negative=False, isinf=True), err=True)
        if clamped.invalid:
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        return _monotonic_incr(OP.log, clamped._lo_endpoint(), clamped._hi_endpoint(), clamped._err, ctxs)

    def log2(self, ctx=None):
        """Computes log2(x) on this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid:
            return Interval(invalid=True)

        clamped = self.clamp(ieee754.Float(0.0), digital.Digital(negative=False, isinf=True), err=True)
        if clamped.invalid:
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        return _monotonic_incr(OP.log2, clamped._lo_endpoint(), clamped._hi_endpoint(), clamped._err, ctxs)

    def log10(self, ctx=None):
        """Computes log10(x) on this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid:
            return Interval(invalid=True)

        clamped = self.clamp(ieee754.Float(0.0), digital.Digital(negative=False, isinf=True), err=True)
        if clamped.invalid:
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        return _monotonic_incr(OP.log10, clamped._lo_endpoint(), clamped._hi_endpoint(), clamped._err, ctxs)

    def log1p(self, ctx=None):
        """Computes log1p(x) on this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid:
            return Interval(invalid=True)

        clamped = self.clamp(ieee754.Float(-1.0), digital.Digital(negative=False, isinf=True), err=True)
        if clamped.invalid:
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        return _monotonic_incr(OP.log1p, clamped._lo_endpoint(), clamped._hi_endpoint(), clamped._err, ctxs)

    def pow(self, other, ctx=None):
        """Computes pow(x, y) on this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid or other._invalid:
            return Interval(invalid=True)

        zero = ieee754.Float(0.0)
        ctxs = self._select_context(self, ctx=ctx)
        if self._hi < zero:
            return _power_neg(self, other, ctxs)
        elif self._lo >= zero:
            return _power_pos(self, other, ctxs)
        else:
            neg, pos = self.split(zero)
            return _power_neg(neg, other).union(_power_pos(pos, other))

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
        try:
            fl = fl_fn(xf, ctx=ctx)
        except gmp.OverflowResultError:
            fl = ieee754.Float('inf', ctx=ctx)
        except gmp.UnderflowResultError:
            fl = ieee754.Float(0.0, ctx=ctx)
            

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
        for xf, yf, fl, ival, err, invalid in bad:
            print(' x={} y={} fl={}, ival={}, err={}, invalid={}'.format(
                str(xf), str(yf), str(fl), str(ival), err, invalid))
    else:
        print('[PASSED] {}'.format(name))

def pow_overflow(x: ieee754.Float, y: ieee754.Float, ctx):
    ctx = x._select_context(x, y, ctx=ctx)
    result = gmpmath.compute(OP.pow, x, y, prec=ctx.p, trap_underflow=False, trap_overflow=False)
    return ieee754.Float._round_to_context(result, ctx)

def test_interval(num_tests=10_000, ctx=ieee754.ieee_ctx(11, 64)):
    """Runs unit tests for the Interval type"""

    # ops = [
    #     ("neg", ieee754.Float.neg, Interval.neg, 1),
    #     ("fabs", ieee754.Float.fabs, Interval.fabs, 1),
    #     ("add", ieee754.Float.add, Interval.add, 2),
    #     ("sub", ieee754.Float.sub, Interval.sub, 2),
    #     ("mul", ieee754.Float.mul, Interval.mul, 2),
    #     ("div", ieee754.Float.div, Interval.div, 2),
    #     ("sqrt", ieee754.Float.sqrt, Interval.sqrt, 1),
    #     ("cbrt", ieee754.Float.cbrt, Interval.cbrt, 1),

    #     ("rint", rint, Interval.rint, 1),
    #     ("round", ieee754.Float.round, Interval.round, 1),
    #     ("ceil", ieee754.Float.ceil, Interval.ceil, 1),
    #     ("floor", ieee754.Float.floor, Interval.floor, 1),
    #     ("trunc", ieee754.Float.trunc, Interval.trunc, 1),

    #     ("exp", ieee754.Float.exp_, Interval.exp, 1),
    #     ("expm1", ieee754.Float.expm1, Interval.expm1, 1),
    #     ("exp2", ieee754.Float.exp2, Interval.exp2, 1),
    #     ("log", ieee754.Float.log, Interval.log, 1),
    #     ("log2", ieee754.Float.log2, Interval.log2, 1),
    #     ("log10", ieee754.Float.log10, Interval.log10, 1),
    #     ("log1p", ieee754.Float.log1p, Interval.log1p, 1),
    #     ("pow", ieee754.Float.pow, Interval.pow, 2),
    # ]

    ops = [
        ("pow", pow_overflow, Interval.pow, 2)
    ]

    random.seed()
    for name, fl_fn, ival_fn, argc in ops:
        if argc == 1:
            test_unary_fn(name, fl_fn, ival_fn, num_tests, ctx)
        elif argc == 2:
            test_binary_fn(name, fl_fn, ival_fn, num_tests, ctx)
        else:
            raise ValueError('invalid argc for testing: argc={}'.format(argc))
