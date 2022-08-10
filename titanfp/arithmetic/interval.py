"""Interval arithmetic based on Rival.
Original implementation: https://github.com/herbie-fp/rival.
"""

from enum import IntEnum, unique
import gmpy2 as gmp
import re

from titanfp.titanic import gmpmath
from . import interpreter
from . import ieee754

from ..titanic import digital
from ..titanic.ops import OP, RM

@unique
class IntervalOrder(IntEnum):
    """Classification of an `Interval` by comparison against a scalar value.
    See `Interval.classify()` for details."""
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

def is_even_integer(x):
    """Takes a Digital type `x` and returns true if `x` is an even integer."""
    if not isinstance(x, digital.Digital):
        raise ValueError('expected a Digital type: {}'.format(x))
    elif not x.is_integer():
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
    elif not x.is_integer():
        return False
    elif x.is_zero() or x.exp > 0:
        return False
    elif -x.exp > x.c.bit_length():
        return False
    else:
        return (abs(x.c) & (2 ** -x.exp)) != 0

#
#   Boolean interval
#

class BooleanInterval(object):
    """A boolean interval based on the Rival interval arithmetic library.
    Original implementation at https://github.com/herbie-fp/rival written by Pavel Panchekha and Oliver Flatt.
    The interval uses boolean bounds, immovability flags to signal a fixed endpoint,
    and error flags to propagate partial or complete domain errors.
    """

    # represents the boolean interval [_lo, _hi]
    # where _lo == _hi means the value is known.
    # (using ieee754.Float since this makes the most sense)
    _lo: bool = False
    _hi: bool = True

    # immovability flags
    # from the original Rival implementation:
    #   "Intervals may shrink (though they cannot grow) when computed at a higher precision.
    #   However, in some cases it is known this will not occur, largely due to overflow.
    #   In those cases, the interval is marked fixed, or immovable."
    #
    _lo_isfixed: bool = False
    _hi_isfixed: bool = True

    # domain error
    _err: bool = False

    # invalid interval
    _invalid: bool = False

    # the internal state is not directly visible: expose it with properties

    @property
    def lo(self):
        """The lower bound.
        """
        return self._lo

    @property
    def hi(self):
        """The upper bound.
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

    def __init__(self,
                 x=None,
                 lo=None,
                 hi=None,
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

        # _lo
        if lo is not None:
            if not isinstance(lo, bool):
                raise ValueError('expected a boolean: lo={}'.format(lo))
            self._lo = lo
        elif x is not None:
            if isinstance(x, BooleanInterval):
                self._lo = x._lo
            elif isinstance(x, bool):
                self._lo = x
            else:
                raise ValueError('expected a boolean: lo={}'.format(lo))
        else:
            self._lo = type(self)._lo
        
        # _hi
        if hi is not None:
            if not isinstance(hi, bool):
                raise ValueError('expected a boolean: hi={}'.format(hi))
            self._hi = hi
        elif x is not None:
            if isinstance(x, BooleanInterval):
                self._hi = x._hi
            elif isinstance(x, bool):
                self._hi = x
            else:
                raise ValueError('expected a boolean: hi={}'.format(hi))
        else:
            self._hi = type(self)._hi  

        # _lo_isfixed
        if lo_isfixed is not None:
            self._lo_isfixed = lo_isfixed
        elif x is not None:
            if isinstance(x, BooleanInterval):
                self._lo_isfixed = x._lo_isfixed
            else:
                self._lo_isfixed = False
        else:
            self._lo_isfixed = type(self)._lo_isfixed

        # _hi_isfixed
        if hi_isfixed is not None:
            self._hi_isfixed = hi_isfixed
        elif x is not None:
            if isinstance(x, BooleanInterval):
                self._hi_isfixed = x._hi_isfixed
            else:
                self._hi_isfixed = False
        else:
            self._hi_isfixed = type(self)._hi_isfixed

        # _err
        if err is not None:
            self._err = err
        elif x is not None:
            if isinstance(x, BooleanInterval):
                self._err = x._err
            else:
                self._err = False
        else:
            self._err = type(self)._err

        # _invalid
        if invalid is not None:
            self._invalid = invalid
        elif x is not None:
            if isinstance(x, BooleanInterval):
                self._invalid = x._invalid
            else:
                self._invalid = self._lo.isnan or self._hi.isnan
        else:
            self._invalid = type(self)._invalid

        # check bounds
        if self._lo and not self._hi:
            raise ValueError('invalid interval: lo={}, hi={}'.format(self._lo, self._hi)) 

    def __repr__(self):
        return '{}(lo={}, hi={}, lo_isfixed={}, hi_isfixed={}, err={}, invalid={})'.format(
            type(self).__name__, repr(self._lo), repr(self._hi), repr(self._lo_isfixed),
            repr(self._hi_isfixed), repr(self._err), repr(self._invalid)
        )

    def __str__(self):
        if self._invalid:
            return '[invalid, invalid]'
        else:
            return '[{}, {}]'.format(str(self._lo), str(self._hi))

    def bool_or_none(self):
        """Returns the boolean representation of this interval if
        both endpoints agree, else returns None."""
        return self._lo if not self._invalid and self._lo == self._hi else None

    def neg(self):
        """Applies boolean NOT this interval."""
        if self._invalid:
            return BooleanInterval(invalid=True)

        lo, lo_isfixed = (not self._hi, self._hi_isfixed)
        hi, hi_isfixed = (not self._lo, self._lo_isfixed)
        err = self._err
        return BooleanInterval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err)

    def conjoin(self, other):
        """Applies boolean AND to this interval and another and returns the result"""
        if self._invalid or other._invalid:
            return BooleanInterval(invalid=True)

        lo = self._lo and other._lo
        hi = self._hi and other._hi
        lo_isfixed = self._lo_isfixed and other._lo_isfixed
        hi_isfixed = self._hi_isfixed and other._hi_isfixed
        err = self._err or other._err
        return BooleanInterval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err)

    def disjoin(self, other):
        """Applies boolean OR to this interval and another and returns the result"""
        if self._invalid or other._invalid:
            return BooleanInterval(invalid=True)

        lo = self._lo or other._lo
        hi = self._hi or other._hi
        lo_isfixed = self._lo_isfixed and other._lo_isfixed
        hi_isfixed = self._hi_isfixed and other._hi_isfixed
        err = self._err or other._err
        return BooleanInterval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err)

    def union(self, other):
        """Returns the union of this interval and another."""
        if self._invalid or other._invalid:
            return BooleanInterval(invalid=True)

        lo = self._lo and other._lo
        hi = self._hi or other._hi
        lo_isfixed = self._lo_isfixed and other._lo_isfixed
        hi_isfixed = self._hi_isfixed and other._hi_isfixed
        err = self._err or other._err
        return BooleanInterval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err)

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
                (a_isfixed and a.isinf and bclass != IntervalOrder.CONTAINS) or \
                (b_isfixed and b.is_zero() and not b.isinf) or \
                (b_isfixed and b.isinf and aclass != IntervalOrder.CONTAINS)
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
                (b_isfixed and b.isinf and aclass != IntervalOrder.CONTAINS)
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
                (a_isfixed and a == digital.Digital(c=0, exp=0) and bclass != IntervalOrder.CONTAINS) or \
                (a_isfixed and a.isinf and bclass != IntervalOrder.CONTAINS) or \
                (b_isfixed and b == digital.Digital(c=0, exp=0)) or \
                (b_isfixed and b.isinf and aclass != IntervalOrder.CONTAINS)
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

# assumes both `x` and `y` are positive
def _fmod_pos(x, y, err, ctxs):
    ctx, lo_ctx, hi_ctx = ctxs
    a = x._lo.div(y._hi, lo_ctx).trunc(lo_ctx)
    b = x._hi.div(y._hi, hi_ctx).trunc(hi_ctx)
    if a == b:
        c = x._hi.div(y._hi, lo_ctx).trunc(lo_ctx)
        d = x._hi.div(y._lo, hi_ctx).trunc(hi_ctx)
        if c == d:
            lo = x._lo.sub(c.mul(y._hi, hi_ctx), lo_ctx)
            hi = x._hi.sub(c.mul(y._lo, lo_ctx), hi_ctx)
            return Interval(lo=lo, hi=hi, err=err, ctx=ctx)
        else:
            lo = ieee754.Float(0.0, ctx)
            hi = x._hi.div(c.add(ieee754.Float(1.0), hi_ctx), hi_ctx)
            return Interval(lo=lo, hi=hi, err=err, ctx=ctx)
    else:
        lo = ieee754.Float(0.0, ctx)
        hi = x._hi
        return Interval(lo=lo, hi=hi, err=err, ctx=ctx)

def _max(x, y):
    return x if x > y else y

def _min(x, y):
    return x if x < y else y

# assumes both `x` and `y` are positive
def _remainder_pos(x, y, err, ctxs):
    ctx, lo_ctx, hi_ctx = ctxs
    a = x._lo.div(y._hi, lo_ctx).round(lo_ctx)
    b = x._hi.div(y._hi, hi_ctx).round(hi_ctx)
    two = ieee754.Float(2.0, ctx)
    if a == b:
        c = x._hi.div(y._hi, lo_ctx).round(lo_ctx)
        d = x._hi.div(y._lo, hi_ctx).round(hi_ctx)
        if c == d:
            y2 = y._hi.div(two, ctx)    # round nearest ??
            lo = _max(x._lo.sub(c.mul(y._hi, hi_ctx), lo_ctx), y2.neg())
            hi = _min(x._hi.sub(c.mul(y._lo, lo_ctx), hi_ctx), y2)
            return Interval(lo=lo, hi=hi, err=err, ctx=ctx)
        else:
            half = ieee754.Float(0.5, ctx)
        y2_lo = _max(x._lo.sub(c.mul(y._hi, hi_ctx), lo_ctx), y._hi.div(two, ctx).neg())
        y2_hi = x._hi.div(c.add(half, lo_ctx), lo_ctx).div(two, ctx)
        return Interval(lo=_min(y2_lo, y2_hi.neg()), hi=y2_hi, err=err, ctx=ctx)
    else:
        y2 = y._hi.div(two, ctx)    # round nearest ??
        return Interval(lo=y2.neg(), hi=y2, err=err, ctx=ctx)

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

    xclass = x.classify(ieee754.Float(1.0))
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
            return Interval(lo=0.0, hi=0.0, lo_isfixed=False, hi_isfixed=False, err=True, ctx=ctx)
        else:
            return Interval(invalid=True)
    elif a == b:
        a_isfixed = y._lo_isfixed and y._hi_isfixed
        p = Interval(lo=a, hi=a, lo_isfixed=a_isfixed, hi_isfixed=a_isfixed, err=err, ctx=ctx)
        if is_odd_integer(a):
            return _power_pos(pos_x, p, ctxs).neg(ctx)
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
        odds = Interval(lo=odd_lo, hi=odd_hi, err=err, ctx=ctx)
        evens = Interval(lo=even_lo, hi=even_hi, err=err, ctx=ctx)
        i1 = _power_pos(pos_x, evens)
        i2 = _power_pos(pos_x, odds).neg(ctx)
        return i1.union(i2)

def _atan2(a, b, c, d, err, ctxs):
    ctx, lo_ctx, hi_ctx = ctxs
    lo, lo_isfixed = _epfn(OP.atan2, a, b, ctx=lo_ctx)
    hi, hi_isfixed = _epfn(OP.atan2, c, d, ctx=hi_ctx)
    return Interval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err, ctx=ctx)

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
    The interval uses MPFR floating-point value bounds, immovability flags to signal a fixed endpoint,
    and error flags to propagate partial or complete domain errors.
    """

    # represents the real number interval [_lo, _hi]
    # (using ieee754.Float since this makes the most sense)
    _lo: ieee754.Float = ieee754.Float(digital.Digital(negative=True, isinf=True), ctx=ieee754.ieee_ctx(11, 64))
    _hi: ieee754.Float = ieee754.Float(digital.Digital(negative=False, isinf=True), ctx=ieee754.ieee_ctx(11, 64))

    # immovability flags
    # from the original Rival implementation:
    #   "Intervals may shrink (though they cannot grow) when computed at a higher precision.
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
        """The upper bound.
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

        # _lo
        if lo is not None:
            self._lo = ieee754.Float(lo, ctx=lo_ctx)
        elif x is not None:
            if isinstance(x, Interval):
                self._lo = ieee754.Float._round_to_context(x._lo, ctx=lo_ctx)
            else:
                if lo is not None:
                    raise ValueError('cannot specify both x={} and [lo={}, hi={}] when x is not an Interval'.format(repr(x), repr(lo), repr(hi)))
                self._lo = ieee754.Float(x, ctx=lo_ctx)
        else:
            self._lo = type(self)._lo
        
        # _hi
        if hi is not None:
            self._hi = ieee754.Float(hi, ctx=hi_ctx)
        elif x is not None:
            if isinstance(x, Interval):
                self._hi = ieee754.Float._round_to_context(x._hi, ctx=hi_ctx)
            else:
                if hi is not None:
                    raise ValueError('cannot specify both x={} and [lo={}, hi={}] when x is not an Interval'.format(repr(x), repr(lo), repr(hi)))
                self._hi = ieee754.Float(x, ctx=lo_ctx)
        else:
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
        if err is not None:
            self._err = err
        elif x is not None:
            if isinstance(x, Interval):
                self._err = x._err
            else:
                self._err = False
        else:
            self._err = type(self)._err

        # _invalid
        if invalid is not None:
            self._invalid = invalid
        elif x is not None:
            if isinstance(x, Interval):
                self._invalid = x._invalid
            else:
                self._invalid = self._lo.isnan or self._hi.isnan
        else:
            self._invalid = type(self)._invalid

        # check bounds
        if not self._invalid and self._lo > self._hi:
            raise ValueError('invalid interval: lo={}, hi={}'.format(self._lo, self._hi)) 

    def __repr__(self):
        return '{}(lo={}, hi={}, lo_isfixed={}, hi_isfixed={}, err={}, invalid={}, ctx={})'.format(
            type(self).__name__, repr(self._lo), repr(self._hi), repr(self._lo_isfixed),
            repr(self._hi_isfixed), repr(self._err), repr(self._invalid), repr(self._ctx)
        )

    def __str__(self):
        if self._invalid:
            return '[nan, nan]'
        else:
            return '[{}, {}]'.format(str(self._lo), str(self._hi))

    def __int__(self):
        return int(self.rint(self.ctx)._lo)

    # (visible) utility functions

    def is_point(self) -> bool:
        """"Is the interval a "singleton" interval, e.g. [a, a]?"""
        return self._lo == self._hi

    def is_integer(self) -> bool:
        """Is this interval an integer?"""
        return self.is_point() and self._lo.is_integer()

    def contains(self, x) -> bool:
        """Does the interval contain `x`?"""
        return self._lo <= x and x <= self._hi

    def classify(self, val=ieee754.Float(0.0), strict=False) -> IntervalOrder:
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

    def compare(self, other) -> bool:
        """Compares this interval returning a 4-tuple of boolean values:
        definitely less than, maybe less than, maybe greater than, definitely greater than.
        All interval comparators can be implemented using this primitive."""
        if not isinstance(other, Interval):
            other = Interval(x=other)
        return self._hi < other._lo, self._lo < other._hi, self._hi > other._lo, self._lo > other._lo

    def union(self, other):
        """Returns the union of this interval and another."""
        if self.invalid or other.invalid:
            return Interval(invalid=True)
        lo, lo_isfixed = self._lo_endpoint() if self._lo < other._lo else other._lo_endpoint()
        hi, hi_isfixed = self._hi_endpoint() if self._hi > other._hi else other._hi_endpoint()
        err = self._err or other._err
        return Interval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err, ctx=self.ctx)

    def clamp(self, lo, hi):
        """Returns a new interval that is clamped between `lo` and `hi`.
        If `err` is `True` and the current interval lies partially outside `[lo, hi]`,
        the `err` field of the new interval will be `true`.
        If `err` is `True` and the current interval lies completely outside `[lo, hi]`,
        the `invalid` field of the new interval will be `true`."""
        if lo > hi or lo.isnan or hi.isnan:
            raise ValueError('invalid clamp bounds: lo={}, hi={}'.format(lo, hi))

        # not a valid interval
        if self._invalid or self._hi < lo or self._lo > hi:
            return Interval(invalid=True)

        lo = self._lo if self._lo > lo else lo
        hi = self._hi if self._hi < hi else hi
        err = self._err or self._lo < lo or self._hi > hi
        return Interval(lo=lo, hi=hi, lo_isfixed=self._lo_isfixed, hi_isfixed=self._hi_isfixed, err=err, ctx=self.ctx)

    def split(self, val):
        """Takes a finite value `val` between `self.lo` and `self.hi` and
        returns two intervals: `[self.lo, val]` and `[val, self.hi]`."""
        if not isinstance(val, digital.Digital):
            raise ValueError('expected a Digital type {}'.format(val))
        if val.isnan or val.isinf:
            raise ValueError('cannot split on a finite value {}'.format(val))
        if val < self._lo or val > self._hi:
            raise ValueError('split value must be between the endpoints {}'.format(val))
        if self.invalid:
            return Interval(invalid=True), Interval(invalid=True)
        return Interval(x=self, hi=val, ctx=self.ctx), Interval(x=self, lo=val, ctx=self.ctx)

    def eq(self, other):
        """Returns a boolean interval representing `self` == `other`."""
        if self._invalid or other._invalid:
            return BooleanInterval(lo=False, hi=False)

        m_lt, c_lt, c_gt, m_gt = self.compare(other)
        lo = not c_lt and c_gt
        hi = not m_lt and m_gt
        isfixed = self._lo_isfixed and self._hi_isfixed and other._lo_isfixed and other._hi_isfixed
        err = self._err or other._err
        return BooleanInterval(lo=lo, hi=hi, lo_isfixed=isfixed, hi_isfixed=isfixed, err=err)

    def lt(self, other):
        """Returns a boolean interval representing `self` < `other`."""
        if self._invalid or other._invalid:
            return BooleanInterval(lo=False, hi=False)

        m_lt, c_lt, _, _ = self.compare(other)
        lo_isfixed = self._hi_isfixed and other._lo_isfixed
        hi_isfixed = self._lo_isfixed and other._hi_isfixed
        err = self._err or other._err
        return BooleanInterval(lo=m_lt, hi=c_lt, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err)

    def lte(self, other):
        """Returns a boolean interval representing `self` <= `other`."""
        if self._invalid or other._invalid:
            return BooleanInterval(lo=False, hi=False)

        _, _, c_gt, m_gt = self.compare(other)
        lo_isfixed = self._hi_isfixed and other._lo_isfixed
        hi_isfixed = self._lo_isfixed and other._hi_isfixed
        err = self._err or other._err
        return BooleanInterval(lo=(not c_gt), hi=(not m_gt), lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err)

    def gt(self, other):
        """Returns a boolean interval representing `self` > `other`."""
        if self._invalid or other._invalid:
            return BooleanInterval(lo=False, hi=False)

        _, _, c_gt, m_gt = self.compare(other)
        lo_isfixed = self._lo_isfixed and other._hi_isfixed
        hi_isfixed = self._hi_isfixed and other._lo_isfixed
        err = self._err or other._err
        return BooleanInterval(lo=m_gt, hi=c_gt, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err)

    def gte(self, other):
        """Returns a boolean interval representing `self` >= `other`."""
        if self._invalid or other._invalid:
            return BooleanInterval(lo=False, hi=False)

        m_lt, c_lt, _, _ = self.compare(other)
        lo_isfixed = self._lo_isfixed and other._hi_isfixed
        hi_isfixed = self._hi_isfixed and other._lo_isfixed
        err = self._err or other._err
        return BooleanInterval(lo=(not c_lt), hi=(not m_lt), lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err)

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

    # classification

    def isfinite(self):
        if self._invalid:
            return Interval(lo=False, hi=False)

        lo = self._lo.isfinite() and self._hi.isfinite()
        hi = self._lo.isfinite() or self._hi.isfinite()
        isfixed = self._lo_isfixed and self._hi_isfixed
        err = self.err
        return BooleanInterval(lo=lo, hi=hi, lo_isfixed=isfixed, hi_isfixed=isfixed, err=err)

    def isinf(self):
        if self._invalid:
            return Interval(lo=False, hi=False)

        lo = self._lo.isinf and self._hi.isinf
        hi = self._lo.isinf or self._hi.isinf
        isfixed = self._lo_isfixed and self._hi_isfixed
        err = self.err
        return BooleanInterval(lo=lo, hi=hi, lo_isfixed=isfixed, hi_isfixed=isfixed, err=err)

    def isnan(self):
        return BooleanInterval(lo=self._invalid, hi=self._invalid)

    def isnormal(self):
        if self._invalid:
            return Interval(lo=False, hi=False)

        lo = self._lo.isnormal() and self._hi.isnormal()
        hi = self._lo.isnormal() or self._hi.isnormal()
        isfixed = self._lo_isfixed and self._hi_isfixed
        err = self.err
        return BooleanInterval(lo=lo, hi=hi, lo_isfixed=isfixed, hi_isfixed=isfixed, err=err)

    def signbit(self):
        if self._invalid:
            return Interval(lo=False, hi=False)

        lo = self._lo.signbit() and self._hi.signbit()
        hi = self._lo.signbit() or self._hi.signbit()
        isfixed = self._lo_isfixed and self._hi_isfixed
        err = self.err
        return BooleanInterval(lo=lo, hi=hi, lo_isfixed=isfixed, hi_isfixed=isfixed, err=err)

    # math operations

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

        if xclass == IntervalOrder.GREATER:
            if yclass == IntervalOrder.GREATER:
                return _multiply(xlo, ylo, xhi, yhi, xclass, yclass, err, ctxs)
            elif yclass == IntervalOrder.LESS:
                return _multiply(xhi, ylo, xlo, yhi, xclass, yclass, err, ctxs)
            else:   # yclass == IntervalOrder.CONTAINS
                return _multiply(xhi, ylo, xhi, yhi, xclass, yclass, err, ctxs)
        elif xclass == IntervalOrder.LESS:
            if yclass == IntervalOrder.GREATER:
                return _multiply(xlo, yhi, xhi, ylo, xclass, yclass, err, ctxs)
            elif yclass == IntervalOrder.LESS:
                return _multiply(xhi, yhi, xlo, ylo, xclass, yclass, err, ctxs)
            else:   # yclass == IntervalOrder.CONTAINS
                return _multiply(xlo, yhi, xlo, ylo, xclass, yclass, err, ctxs)
        else:   # xclass == IntervalOrder.CONTAINS
            if yclass == IntervalOrder.GREATER:
                return _multiply(xlo, yhi, xhi, yhi, xclass, yclass, err, ctxs)
            elif yclass == IntervalOrder.LESS:
                return _multiply(xhi, ylo, xlo, ylo, xclass, yclass, err, ctxs)
            else:   # yclass == IntervalOrder.CONTAINS
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
               (self.classify(strict=True) == IntervalOrder.CONTAINS and \
                other.classify(strict=True) == IntervalOrder.CONTAINS)

        if yclass == IntervalOrder.CONTAINS:
            # result is split, so we give up and always return [-inf, inf]
            isfixed = self._lo_isfixed and other._lo_isfixed
            return Interval(lo_isfixed=isfixed, hi_isfixed=isfixed, ctx=ctxs[0])
        elif xclass == IntervalOrder.GREATER:
            if yclass == IntervalOrder.GREATER:
                return _divide(xlo, yhi, xhi, ylo, xclass, err, ctxs)
            else:   # yclass == IntervalOrder.LESS
                return _divide(xhi, yhi, xlo, ylo, xclass, err, ctxs)
        elif xclass == IntervalOrder.LESS:
            if yclass == IntervalOrder.GREATER:
                return _divide(xlo, ylo, xhi, yhi, xclass, err, ctxs)
            else:   # yclass == IntervalSign.LESS
                return _divide(xhi, ylo, xlo, yhi, xclass, err, ctxs)
        else:   # xclass == IntervalOrder.GREATER
            if yclass == IntervalOrder.GREATER:
                return _divide(xlo, ylo, xhi, ylo, xclass, err, ctxs)
            else:   # yclass == IntervalOrder.LESS
                return _divide(xhi, yhi, xlo, yhi, xclass, err, ctxs)
    
    def fabs(self, ctx=None):
        """Returns the absolute value of this interval.
        The precision of the interval can be specified by `ctx`.
        """
        if self._invalid:
            return Interval(invalid=True)

        ctx, lo_ctx, hi_ctx = self._select_context(self, ctx=ctx)
        cl = self.classify()
        if cl == IntervalOrder.GREATER:
            return Interval(x=self)
        elif cl == IntervalOrder.LESS:
            return self.neg(ctx)
        else:
            neg_lo, _ = _epfn(OP.neg, self._lo_endpoint(), ctx=hi_ctx)
            lo, lo_isfixed = ieee754.Float(0, ctx=lo_ctx), self._lo_isfixed and self._hi_isfixed
            hi, hi_isfixed = (neg_lo, self._lo_isfixed) if neg_lo > self._hi else (self._hi, self._hi_isfixed)
            return Interval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=self.err, ctx=ctx)

    def hypot(self, other, ctx=None):
        """Performs sqrt(x^2 + y^2) on this interval and another.
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

        clamped = self.clamp(ieee754.Float(0.0), digital.Digital(negative=False, isinf=True))
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

    def fmin(self, other, ctx=None):
        """Performs fmin(x, y) on this interval and another.
        The precision of the interval can be specified by `ctx`.
        """
        if self._invalid or other._invalid:
            return Interval(invalid=True)

        ctx, _, _ = self._select_context(self, ctx=ctx)
        lo, lo_isfixed = (self._lo, self._lo_isfixed) if self._lo < other._lo else (other._lo, other._lo_isfixed)
        hi, hi_isfixed = (self._hi, self._hi_isfixed) if self._hi < other._hi else (other._hi, other._hi_isfixed)
        err = self.err or other.err
        return Interval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err, ctx=ctx)

    def fmax(self, other, ctx=None):
        """Performs fmax(x, y) on this interval and another.
        The precision of the interval can be specified by `ctx`.
        """
        if self._invalid or other._invalid:
            return Interval(invalid=True)

        ctx, _, _ = self._select_context(self, ctx=ctx)
        lo, lo_isfixed = (self._lo, self._lo_isfixed) if self._lo > other._lo else (other._lo, other._lo_isfixed)
        hi, hi_isfixed = (self._hi, self._hi_isfixed) if self._hi > other._hi else (other._hi, other._hi_isfixed)
        err = self.err or other.err
        return Interval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err, ctx=ctx)

    def copysign(self, other, ctx):
        """Performs copysign(x, y) on this interval and another.
        The precision of the interval can be specified by `ctx`.
        """
        if self.invalid or other.invalid:
            return Interval(invalid=True)

        magnitude = self.fabs()
        can_neg = other._lo.negative
        can_pos = not other._hi.negative
        err = self.err or self.err

        ctx, lo_ctx, hi_ctx = self._select_context(magnitude, ctx=ctx)
        if can_neg and can_pos:
            lo, lo_isfixed = _epfn(OP.neg, (magnitude._hi, magnitude._hi_isfixed), ctx=lo_ctx)
            return Interval(lo=lo, hi=magnitude.hi, lo_isfixed=lo_isfixed, hi_isfixed=magnitude.hi_isfixed, err=err, ctx=ctx)
        elif can_neg and not can_pos:
            lo, lo_isfixed = _epfn(OP.neg, (magnitude._hi, magnitude._hi_isfixed), ctx=lo_ctx)
            hi, hi_isfixed = _epfn(OP.neg, (magnitude._lo, magnitude._lo_isfixed), ctx=hi_ctx)
            return Interval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=err, ctx=ctx)
        else:
            return Interval(x=magnitude)

    def fdim(self, other, ctx=None):
        """Performs fmax(x - y, 0) on this interval and another.
        The precision of the interval can be specified by `ctx`.
        """
        zero = Interval(lo=ieee754.Float(0.0), hi=ieee754.Float(0.0))
        return self.sub(other, ctx).fmax(zero, ctx)

    def fmod(self, other, ctx=None):
        """Computes fmod(x, y) on this interval and another.
        The precision of the interval can be specified by `ctx`."""
        zero = ieee754.Float(0.0)
        if self.invalid or other.invalid or (other._lo == zero and other.is_point()):
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        err = self.err or other.err or other.contains(zero)
        pos_y = other.fabs(other.ctx)
        if self._hi <= zero:
            return _fmod_pos(self.neg(self.ctx), pos_y, err, ctxs).neg(ctx)
        elif self._lo >= zero:
            return _fmod_pos(self, pos_y, err, ctxs)
        else:
            neg, pos = self.split(zero)
            return _fmod_pos(pos, pos_y, err, ctxs).union(_fmod_pos(neg.neg(neg.ctx), pos_y, err, ctxs).neg(ctx))

    def remainder(self, other, ctx=None):
        """Computes remainder(x, y) on this interval and another.
        The precision of the interval can be specified by `ctx`."""
        zero = ieee754.Float(0.0)
        if self.invalid or other.invalid or (other._lo == zero and other.is_point()):
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        err = self.err or other.err or other.contains(zero)
        pos_y = other.fabs(other.ctx)
        if self._hi <= zero:
            return _remainder_pos(self.neg(self.ctx), pos_y, err, ctxs).neg(ctx)
        elif self._lo >= zero:
            return _remainder_pos(self, pos_y, err, ctxs)
        else:
            neg, pos = self.split(zero)
            return _remainder_pos(pos, pos_y, err, ctxs).union(_remainder_pos(neg.neg(neg.ctx), pos_y, err, ctxs).neg(ctx))

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

    def nearbyint(self, ctx=None):
        """Rounds this interval to a nearby integer.
        Normally, `nearbyint` respects rounding mode, but interval arithmetic will round conservatively.
        The precision of the interval can be specified by `ctx`."""
        # no different than `rint` except for flags, so ...
        return self.rint(ctx)

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

        clamped = self.clamp(ieee754.Float(0.0), digital.Digital(negative=False, isinf=True))
        if clamped.invalid:
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        return _monotonic_incr(OP.log, clamped._lo_endpoint(), clamped._hi_endpoint(), clamped._err, ctxs)

    def log2(self, ctx=None):
        """Computes log2(x) on this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid:
            return Interval(invalid=True)

        clamped = self.clamp(ieee754.Float(0.0), digital.Digital(negative=False, isinf=True))
        if clamped.invalid:
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        return _monotonic_incr(OP.log2, clamped._lo_endpoint(), clamped._hi_endpoint(), clamped._err, ctxs)

    def log10(self, ctx=None):
        """Computes log10(x) on this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid:
            return Interval(invalid=True)

        clamped = self.clamp(ieee754.Float(0.0), digital.Digital(negative=False, isinf=True))
        if clamped.invalid:
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        return _monotonic_incr(OP.log10, clamped._lo_endpoint(), clamped._hi_endpoint(), clamped._err, ctxs)

    def log1p(self, ctx=None):
        """Computes log1p(x) on this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid:
            return Interval(invalid=True)

        clamped = self.clamp(ieee754.Float(-1.0), digital.Digital(negative=False, isinf=True))
        if clamped.invalid:
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        return _monotonic_incr(OP.log1p, clamped._lo_endpoint(), clamped._hi_endpoint(), clamped._err, ctxs)

    def logb(self, ctx=None):
        """Computes logb(x) on this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid:
            return Interval(invalid=True)

        pos = self.fabs()
        lpos = pos.log()
        return lpos.floor(ctx)

    def pow(self, other, ctx=None):
        """Computes pow(x, y) for this interval and another and returns the result.
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
            return _power_neg(neg, other, ctxs).union(_power_pos(pos, other, ctxs))

    def sin(self, ctx=None):
        """Computes sin(x) for this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid:
            return Interval(invalid=True)
        
        ctx, lo_ctx, hi_ctx = self._select_context(self, ctx=ctx)
        lo_pi = ieee754.Float._round_to_context(gmpmath.compute_constant('PI'), lo_ctx)
        hi_pi = ieee754.Float._round_to_context(gmpmath.compute_constant('PI'), hi_ctx)
        one = ieee754.Float(1.0)
        half = ieee754.Float(0.5)
        zero = ieee754.Float(0.0)

        a = self._lo.div(lo_pi if self._lo < zero else hi_pi, lo_ctx).sub(half, lo_ctx).floor(lo_ctx)
        b = self._hi.div(hi_pi if self._hi < zero else lo_pi, hi_ctx).sub(half, hi_ctx).floor(hi_ctx)
        if a == b:
            if is_even_integer(a):
                lo, lo_isfixed = _epfn(OP.sin, self._hi_endpoint(), ctx=lo_ctx)
                hi, hi_isfixed = _epfn(OP.sin, self._lo_endpoint(), ctx=hi_ctx)
                return Interval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=self.err, ctx=ctx)
            else:
                lo, lo_isfixed = _epfn(OP.sin, self._lo_endpoint(), ctx=lo_ctx)
                hi, hi_isfixed = _epfn(OP.sin, self._hi_endpoint(), ctx=hi_ctx)
                return Interval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=self.err, ctx=ctx)
        elif b.sub(a) == one:
            if is_even_integer(a):
                hi1, hi1_isfixed = _epfn(OP.sin, self._lo_endpoint(), ctx=hi_ctx)
                hi2, hi2_isfixed = _epfn(OP.sin, self._hi_endpoint(), ctx=hi_ctx)
                hi, hi_isfixed = (hi1, hi1_isfixed) if hi1 > hi2 else (hi2, hi2_isfixed)
                return Interval(lo=one.neg(ctx), hi=hi, lo_isfixed=False, hi_isfixed=hi_isfixed, err=self.err, ctx=ctx)
            else:
                lo1, lo1_isfixed = _epfn(OP.sin, self._lo_endpoint(), ctx=lo_ctx)
                lo2, lo2_isfixed = _epfn(OP.sin, self._hi_endpoint(), ctx=lo_ctx)
                lo, lo_isfixed = (lo1, lo1_isfixed) if lo1 < lo2 else (lo2, lo2_isfixed)
                return Interval(lo=lo, hi=one, lo_isfixed=lo_isfixed, hi_isfixed=False, err=self.err, ctx=ctx)
        else:
            return Interval(lo=ieee754.Float(-1), hi=ieee754.Float(1), err=self.err, ctx=ctx)
    
    def cos(self, ctx=None):
        """Computes cos(x) for this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid:
            return Interval(invalid=True)
        
        ctx, lo_ctx, hi_ctx = self._select_context(self, ctx=ctx)
        lo_pi = ieee754.Float._round_to_context(gmpmath.compute_constant('PI'), lo_ctx)
        hi_pi = ieee754.Float._round_to_context(gmpmath.compute_constant('PI'), hi_ctx)
        one = ieee754.Float(1.0)
        zero = ieee754.Float(0.0)

        a = self._lo.div(lo_pi if self._lo < zero else hi_pi, lo_ctx).floor(lo_ctx)
        b = self._hi.div(hi_pi if self._hi < zero else lo_pi, hi_ctx).floor(hi_ctx)
        if a == b:
            if is_even_integer(a):
                lo, lo_isfixed = _epfn(OP.cos, self._hi_endpoint(), ctx=lo_ctx)
                hi, hi_isfixed = _epfn(OP.cos, self._lo_endpoint(), ctx=hi_ctx)
                return Interval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=self.err, ctx=ctx)
            else:
                lo, lo_isfixed = _epfn(OP.cos, self._lo_endpoint(), ctx=lo_ctx)
                hi, hi_isfixed = _epfn(OP.cos, self._hi_endpoint(), ctx=hi_ctx)
                return Interval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, err=self.err, ctx=ctx)
        elif b.sub(a) == one:
            if is_even_integer(a):
                hi1, hi1_isfixed = _epfn(OP.cos, self._lo_endpoint(), ctx=hi_ctx)
                hi2, hi2_isfixed = _epfn(OP.cos, self._hi_endpoint(), ctx=hi_ctx)
                hi, hi_isfixed = (hi1, hi1_isfixed) if hi1 > hi2 else (hi2, hi2_isfixed)
                return Interval(lo=one.neg(ctx), hi=hi, lo_isfixed=False, hi_isfixed=hi_isfixed, err=self.err, ctx=ctx)
            else:
                lo1, lo1_isfixed = _epfn(OP.cos, self._lo_endpoint(), ctx=lo_ctx)
                lo2, lo2_isfixed = _epfn(OP.cos, self._hi_endpoint(), ctx=lo_ctx)
                lo, lo_isfixed = (lo1, lo1_isfixed) if lo1 < lo2 else (lo2, lo2_isfixed)
                return Interval(lo=lo, hi=one, lo_isfixed=lo_isfixed, hi_isfixed=False, err=self.err, ctx=ctx)
        else:
            return Interval(lo=ieee754.Float(-1), hi=ieee754.Float(1), err=self.err, ctx=ctx)

    def tan(self, ctx=None):
        """Computes tan(x) for this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid:
            return Interval(invalid=True)
        
        ctx, lo_ctx, hi_ctx = self._select_context(self, ctx=ctx)
        lo_pi = ieee754.Float._round_to_context(gmpmath.compute_constant('PI'), lo_ctx)
        hi_pi = ieee754.Float._round_to_context(gmpmath.compute_constant('PI'), hi_ctx)
        half = ieee754.Float(0.5)
        zero = ieee754.Float(0.0)

        a = self._lo.div(lo_pi if self._lo < zero else hi_pi, lo_ctx).sub(half, lo_ctx).floor(lo_ctx)
        b = self._hi.div(hi_pi if self._hi < zero else lo_pi, hi_ctx).sub(half, hi_ctx).floor(hi_ctx)
        if a == b:
            lo, lo_isfixed = _epfn(OP.tan, self._lo_endpoint(), ctx=lo_ctx)
            hi, hi_isfixed = _epfn(OP.tan, self._hi_endpoint(), ctx=hi_ctx)
            return Interval(lo=lo, hi=hi, lo_isfixed=lo_isfixed, hi_isfixed=hi_isfixed, ctx=ctx)
        else:
            return Interval(invalid=True)

    def asin(self, ctx=None):
        """Computes arcsin(x) on this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid:
            return Interval(invalid=True)

        one = ieee754.Float(1.0)
        ctxs = self._select_context(self, ctx=ctx)
        clamped = self.clamp(one.neg(ctxs[0]), one)
        if clamped.invalid:
            return Interval(invalid=True)

        return _monotonic_incr(OP.asin, clamped._lo_endpoint(), clamped._hi_endpoint(), clamped._err, ctxs)

    def acos(self, ctx=None):
        """Computes arccos(x) on this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid:
            return Interval(invalid=True)

        one = ieee754.Float(1.0)
        ctxs = self._select_context(self, ctx=ctx)
        clamped = self.clamp(one.neg(ctxs[0]), one)
        if clamped.invalid:
            return Interval(invalid=True)

        return _monotonic_decr(OP.acos, clamped._lo_endpoint(), clamped._hi_endpoint(), clamped._err, ctxs)

    def atan(self, ctx=None):
        """Computes arctan(x) on this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid:
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        return _monotonic_incr(OP.atan, self._lo_endpoint(), self._hi_endpoint(), self._err, ctxs)

    def atan2(self, other, ctx=None):
        """Computes atan2(y, x) on this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid or other._invalid:
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        err = self.err or other.err
        zero = ieee754.Float(0.0)
    
        yclass = self.compare(zero)
        yhi = self._hi_endpoint()
        ylo = self._lo_endpoint()

        xclass = other.compare(zero)
        xhi = other._hi_endpoint()
        xlo = other._lo_endpoint()

        if xclass == IntervalOrder.GREATER and yclass == IntervalOrder.GREATER:
            return _atan2(ylo, xhi, yhi, xlo, err, ctxs)
        elif xclass == IntervalOrder.GREATER and yclass == IntervalOrder.CONTAINS:
            return _atan2(ylo, xlo, yhi, xlo, err, ctxs)
        elif xclass == IntervalOrder.GREATER and yclass == IntervalOrder.LESS:
            return _atan2(ylo, xlo, yhi, xhi, err, ctxs)
        elif xclass == IntervalOrder.CONTAINS and yclass == IntervalOrder.GREATER:
            return _atan2(ylo, xhi, ylo, xlo, err, ctxs)
        elif xclass == IntervalOrder.CONTAINS and yclass == IntervalOrder.LESS:
            return _atan2(yhi, xlo, yhi, xhi, err, ctxs)
        elif xclass == IntervalOrder.LESS and yclass == IntervalOrder.GREATER:
            return _atan2(yhi, xhi, ylo, xlo, err, ctxs)
        elif xclass == IntervalOrder.LESS and yclass == IntervalOrder.LESS:
            return _atan2(yhi, xlo, ylo, xhi, err, ctxs)
        else:
            ctx, lo_ctx, hi_ctx = self._select_context(self, ctx=ctx)
            lo_pi = ieee754.Float._round_to_context(gmpmath.compute_constant('PI'), lo_ctx)
            hi_pi = ieee754.Float._round_to_context(gmpmath.compute_constant('PI'), hi_ctx)
            err = err or other._hi >= zero
            invalid = other._lo == zero and other._hi == zero and self._lo == zero and self._hi == zero
            return Interval(lo=lo_pi.neg(ctx), hi=hi_pi, err=err, invalid=invalid, ctx=ctx)
    
    def sinh(self, ctx=None):
        """Computes sinh(x) on this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid:
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        return _monotonic_incr(OP.sinh, self._lo_endpoint(), self._hi_endpoint(), self._err, ctxs)

    def cosh(self, ctx=None):
        """Computes cosh(x) on this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid:
            return Interval(invalid=True)

        pos = self.fabs(self.ctx)
        ctxs = self._select_context(self, ctx=ctx)
        return _monotonic_incr(OP.cosh, pos._lo_endpoint(), pos._hi_endpoint(), pos._err, ctxs)

    def tanh(self, ctx=None):
        """Computes tanh(x) on this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid:
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        return _monotonic_incr(OP.tanh, self._lo_endpoint(), self._hi_endpoint(), self._err, ctxs)
    
    def asinh(self, ctx=None):
        """Computes asinh(x) on this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid:
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        return _monotonic_incr(OP.asinh, self._lo_endpoint(), self._hi_endpoint(), self._err, ctxs)

    def acosh(self, ctx=None):
        """Computes acosh(x) on this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid:
            return Interval(invalid=True)

        one = ieee754.Float(1.0)
        inf = digital.Digital(isinf=True)
        clamped = self.clamp(one, inf)
        if clamped.invalid:
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        return _monotonic_incr(OP.acosh, clamped._lo_endpoint(), clamped._hi_endpoint(), clamped._err, ctxs)

    def atanh(self, ctx=None):
        """Computes atanh(x) on this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid:
            return Interval(invalid=True)

        one = ieee754.Float(1.0)
        clamped = self.clamp(one.neg(), one)
        if clamped.invalid:
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        return _monotonic_incr(OP.atanh, clamped._lo_endpoint(), clamped._hi_endpoint(), clamped._err, ctxs)

    def erf(self, ctx=None):
        """Computes erf(x) on this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid:
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        return _monotonic_incr(OP.erf, self._lo_endpoint(), self._hi_endpoint(), self._err, ctxs)

    def erfc(self, ctx=None):
        """Computes erfc(x) on this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        if self._invalid:
            return Interval(invalid=True)

        ctxs = self._select_context(self, ctx=ctx)
        return _monotonic_decr(OP.erfc, self._lo_endpoint(), self._hi_endpoint(), self._err, ctxs)

    def lgamma(self, ctx=None):
        """Computes lgamma(x) on this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        raise ValueError('unimplemented method')

    def tgamma(self, ctx=None):
        """Computes tgamma(x) on this interval and returns the result.
        The precision of the interval can be specified by `ctx`"""
        raise ValueError('unimplemented method')

#
#   Interpreter
#

class Interpreter(interpreter.StandardInterpreter):
    dtype = Interval
    ctype = ieee754.IEEECtx

    def arg_to_digital(self, x, ctx):
        if isinstance(x, str):
            m = re.match(r'\[(?P<lo>.*),(?P<hi>.*)\]', x)
            if m is not None:
                md = m.groupdict()
                return self.dtype(lo=md['lo'], hi=md['hi'], ctx=ctx)
            else:
                return self.dtype(x, ctx=ctx)
        else:
            return self.dtype(x, ctx=ctx)

    def round_to_context(self, x, ctx):
        return self.dtype(x, ctx=ctx)

    def _eval_constant(self, e, ctx):
        try:
            return None, self.dtype(x=self.constants[e.value], ctx=ctx)
        except KeyError:
            return None, self.round_to_context(gmpmath.compute_constant(e.value, prec=ctx.p), ctx=ctx)

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

    def _eval_isinf(self, e, ctx):
        in0 = self.evaluate(e.children[0], ctx)
        if isinstance(in0, BooleanInterval):
            return [in0], BooleanInterval(lo=False, hi=False)
        else:
            return [in0], in0.isinf()

    def _eval_isnan(self, e, ctx):
        in0 = self.evaluate(e.children[0], ctx)
        if isinstance(in0, BooleanInterval):
            return [in0], BooleanInterval(lo=False, hi=False)
        else:
            return [in0], in0.isnan()

    def _eval_lt(self, e, ctx):
        if len(e.children) == 2:
            in0 = self.evaluate(e.children[0], ctx)
            in1 = self.evaluate(e.children[1], ctx)
            return [in0, in1], in0.lt(in1)
        elif len(e.children) < 2:
            return [], BooleanInterval(x=True)
        else:
            inputs = []
            result = BooleanInterval(x=True)
            a = self.evaluate(e.children[0], ctx)
            inputs.append(a)
            for child in e.children[1:]:
                b = self.evaluate(child, ctx)
                inputs.append(b)
                result = result.conjoin(a.lt(b))
                if result.bool_or_none() == False:
                    return inputs, BooleanInterval(x=False)
                a = b
            return [], result

    def _eval_gt(self, e, ctx):
        if len(e.children) == 2:
            in0 = self.evaluate(e.children[0], ctx)
            in1 = self.evaluate(e.children[1], ctx)
            return [in0, in1], in0.gt(in1)
        elif len(e.children) < 2:
            return [], BooleanInterval(x=True)
        else:
            inputs = []
            result = BooleanInterval(x=True)
            a = self.evaluate(e.children[0], ctx)
            inputs.append(a)
            for child in e.children[1:]:
                b = self.evaluate(child, ctx)
                inputs.append(b)
                result = result.conjoin(a.gt(b))
                if result.bool_or_none() == False:
                    return inputs, BooleanInterval(x=False)
                a = b
            return [], result

    def _eval_lte(self, e, ctx):
        if len(e.children) == 2:
            in0 = self.evaluate(e.children[0], ctx)
            in1 = self.evaluate(e.children[1], ctx)
            return [in0, in1], in0.lte(in1)
        elif len(e.children) < 2:
            return [], BooleanInterval(x=True)
        else:
            inputs = []
            result = BooleanInterval(x=True)
            a = self.evaluate(e.children[0], ctx)
            inputs.append(a)
            for child in e.children[1:]:
                b = self.evaluate(child, ctx)
                inputs.append(b)
                result = result.conjoin(a.lte(b))
                if result.bool_or_none() == False:
                    return inputs, BooleanInterval(x=False)
                a = b
            return [], result

    def _eval_gte(self, e, ctx):
        if len(e.children) == 2:
            in0 = self.evaluate(e.children[0], ctx)
            in1 = self.evaluate(e.children[1], ctx)
            return [in0, in1], in0.gte(in1)
        elif len(e.children) < 2:
            return [], BooleanInterval(x=True)
        else:
            inputs = []
            result = BooleanInterval(x=True)
            a = self.evaluate(e.children[0], ctx)
            inputs.append(a)
            for child in e.children[1:]:
                b = self.evaluate(child, ctx)
                inputs.append(b)
                result = result.conjoin(a.gte(b))
                if result.bool_or_none() == False:
                    return inputs, BooleanInterval(x=False)
                a = b
            return [], result

    def _eval_eq(self, e, ctx):
        if len(e.children) == 2:
            in0 = self.evaluate(e.children[0], ctx)
            in1 = self.evaluate(e.children[1], ctx)
            return [in0, in1], in0.eq(in1)
        elif len(e.children) < 2:
            return [], BooleanInterval(x=True)
        else:
            inputs = []
            result = BooleanInterval(x=True)
            a = self.evaluate(e.children[0], ctx)
            inputs.append(a)
            for child in e.children[1:]:
                b = self.evaluate(child, ctx)
                inputs.append(b)
                result = result.conjoin(a.eq(b))
                if result.bool_or_none() == False:
                    return inputs, BooleanInterval(x=False)
                a = b
            return [], result

    def _eval_neq(self, e, ctx):
        if len(e.children) == 2:
            in0 = self.evaluate(e.children[0], ctx)
            in1 = self.evaluate(e.children[1], ctx)
            return [in0, in1], in0.neq(in1)
        elif len(e.children) < 2:
            return [], BooleanInterval(x=True)
        else:
            inputs = []
            result = BooleanInterval(x=True)
            a = self.evaluate(e.children[0], ctx)
            inputs.append(a)
            for child in e.children[1:]:
                b = self.evaluate(child, ctx)
                inputs.append(b)
                result = result.conjoin(a.eq(b))
                if result.bool_or_none() == True:
                    return inputs, BooleanInterval(x=False)
                a = b
            return [], result.neg()

    def _eval_and(self, e, ctx):
        result = BooleanInterval(x=True)
        for child in e.children:
            result = result.conjoin(self.evaluate(child, ctx))
        return None, result

    def _eval_or(self, e, ctx):
        result = BooleanInterval(x=True)
        for child in e.children:
            result = result.disjoin(self.evaluate(child, ctx))
        return None, result

    def _eval_not(self, e, ctx):
        return None, self.evaluate(e.children[0], ctx).negate()

    def _eval_if(self, e, ctx):
        cond = self.evaluate(e.cond, ctx)
        if cond.invalid:    # this seems to be what Titanic normally
            result = Interval(invalid=True)
        elif cond.lo:
            ift = self.evaluate(e.then_body, ctx)
            if ift.invalid:
                result = Interval(invalid=True)
            else:
                err = ift.err and cond.err
                result = Interval(x=ift, err=err)
        elif not cond.hi:
            iff = self.evaluate(e.else_body, ctx)
            if iff.invalid:
                result = Interval(invalid=True)
            else:
                err = iff.err and cond.err
                result = Interval(x=iff, err=err)
        else:
            ift = self.evaluate(e.then_body, ctx)
            iff = self.evaluate(e.else_body, ctx)
            u = ift.union(iff)
            err = u.err and cond.err
            result = Interval(x=u, err=err)
        return None, result

#
#   Testing
#

import random

class Ordinal(object):
    def __init__(self, v, ctx):
        if isinstance(v, ieee754.Float):
            bits = ieee754.digital_to_bits(v, ctx=ctx)
            sign = bits >> (ctx.nbits - 1)
            mag = bits & ((1 << (ctx.nbits - 1)) - 1)
            self.ord = -mag if sign == 1 else mag
            self.f = v
        elif isinstance(v, int):
            self.f = ieee754.bits_to_digital(((1 << (ctx.nbits - 1)) + abs(v)) if v < 0 else v)
            self.ord = v
        else:
            raise ValueError('Cannot convert {} to an Ordinal type'.format(v))

    def __repr__(self):
        return '{}(v={}) # ord={}, fl={}'.format(type(self).__name__, self.ord, self.ord, self.f)

    def __str__(self):
        return '{}(ord={}, fl={})'.format(type(self).__name__, self.ord, self.f)


def random_float(min=None, max=None, ctx=None):
    if ctx is None:
        ctx = ieee754.ieee_ctx(11, 64)
    inf_ord = Ordinal(ieee754.Float('inf', ctx=ctx), ctx=ctx).ord

    if min is not None:
        min_ord = Ordinal(min, ctx=ctx).ord
    else:
        min_ord = 1 - inf_ord
    
    if max is not None:
        max_ord = Ordinal.digital_to_bits(max, ctx=ctx).ord
    else:
        max_ord = inf_ord - 1

    ord = random.randint(min_ord, max_ord)
    return Ordinal(ord, ctx=ctx).f

def random_wide_interval(min=None, max=None, ctx=None):
    if ctx is None:
        ctx = ieee754.ieee_ctx(11, 64)

    v1 = random_float(min, max, ctx)
    v2 = random_float(min, max, ctx)
    if v1 > v2:
        return Interval(lo=v2, hi=v1)
    else:
        return Interval(lo=v1, hi=v2)

def random_narrow_interval(min=None, max=None, ctx=None):
    if ctx is None:
        ctx = ieee754.ieee_ctx(11, 64)

    v1 = random_float(min, max, ctx)
    delt = random.randint(1, ctx.p)
    sdelt = delt * (-1 if random.getrandbits(1) == 1 else 1)
    v2 = Ordinal(Ordinal(v1, ctx=ctx).ord + sdelt, ctx=ctx).f
    if sdelt < 0:
        return Interval(lo=v2, hi=v1)
    else:
        return Interval(lo=v1, hi=v2)

def random_interval(min=None, max=None, ctx=None):
    type = random.randint(1, 3)
    if type == 1:
        return random_wide_interval(min, max, ctx)
    elif type == 2:
        return random_narrow_interval(min, max, ctx)
    else:
        fl = random_float(min, max, ctx)
        return Interval(lo=fl, hi=fl)

def sample_interval(ival, ctx=None):
    if ctx is None:
        ctx = ieee754.ieee_ctx(11, 64)

    lo = Ordinal(ival.lo, ctx=ctx).ord
    hi = Ordinal(ival.hi, ctx=ctx).ord
    return Ordinal(random.randint(lo, hi), ctx=ctx).f

def test_unary_fn(name, fl_fn, ival_fn, num_tests, ctx, verbose):
    bad = []
    for _ in range(num_tests):
        x = random_interval(ctx=ctx)
        ival = ival_fn(x, ctx=ctx)

        xf = sample_interval(x, ctx)
        try:
            fl = fl_fn(xf, ctx=ctx)
        except gmp.OverflowResultError:
            fl = ieee754.Float('inf', ctx=ctx)
        except gmp.UnderflowResultError:
            fl = ieee754.Float(0.0, ctx=ctx)

        if fl.isnan:
            if x.is_point() and not ival.invalid:
                bad.append((x, xf, fl, ival, ival.err, ival.invalid))
            elif not ival.err and not ival.invalid:
                bad.append((x, xf, fl, ival, ival.err, ival.invalid))
        elif fl < ival.lo or fl > ival.hi:
            bad.append((x, xf, fl, ival, ival.err, ival.invalid))
    if len(bad) > 0:
        print('[FAILED] {} {}/{}'.format(name, len(bad), num_tests))
        if verbose:
            for x, xf, fl, ival, err, invalid in bad:
                print(' x={} ({}) fl={}, ival={}, err={}, invalid={}'.format(
                    str(x), str(xf), str(fl), str(ival), err, invalid))
    else:
        print('[PASSED] {}'.format(name))

def test_binary_fn(name, fl_fn, ival_fn, num_tests, ctx, verbose):
    bad = []
    for _ in range(num_tests):
        x = random_interval(ctx=ctx)
        y = random_interval(ctx=ctx)
        ival = ival_fn(x, y, ctx=ctx)

        xf = sample_interval(x, ctx=ctx)
        yf = sample_interval(y, ctx=ctx)
        try:
            fl = fl_fn(xf, yf, ctx=ctx)
        except gmp.OverflowResultError:
            fl = ieee754.Float('inf', ctx=ctx)
        except gmp.UnderflowResultError:
            fl = ieee754.Float(0.0, ctx=ctx) 

        if fl.isnan:
            if x.is_point() and y.is_point() and not ival.invalid:
                bad.append((x, y, xf, yf, fl, ival, ival.err, ival.invalid))
            elif not ival.err and not ival.invalid:
                bad.append((x, y, xf, yf, fl, ival, ival.err, ival.invalid))
        elif  fl < ival.lo or fl > ival.hi:
            bad.append((x, y, xf, yf, fl, ival, ival.err, ival.invalid))
    if len(bad) > 0:
        print('[FAILED] {} {}/{}'.format(name, len(bad), num_tests))
        if verbose:
            for x, y, xf, yf, fl, ival, err, invalid in bad:
                print(' x={} ({}) y={} ({}) fl={}, ival={}, err={}, invalid={}'.format(
                    str(x), str(xf), str(y), str(yf), str(fl), str(ival), err, invalid))
    else:
        print('[PASSED] {}'.format(name))

def test_ternary_fn(name, fl_fn, ival_fn, num_tests, ctx, verbose):
    bad = []
    for _ in range(num_tests):
        x = random_interval(ctx=ctx)
        y = random_interval(ctx=ctx)
        z = random_interval(ctx=ctx)
        ival = ival_fn(x, y, z, ctx=ctx)

        xf = sample_interval(x, ctx=ctx)
        yf = sample_interval(y, ctx=ctx)
        zf = sample_interval(z, ctx=ctx)
        try:
            fl = fl_fn(xf, yf, zf, ctx=ctx)
        except gmp.OverflowResultError:
            fl = ieee754.Float('inf', ctx=ctx)
        except gmp.UnderflowResultError:
            fl = ieee754.Float(0.0, ctx=ctx)

        if fl.isnan:
            if x.is_point() and y.is_point() and z.is_point() and not ival.invalid:
                bad.append((x, y, z, xf, yf, zf, fl, ival, ival.err, ival.invalid))
            elif not ival.err and not ival.invalid:
                bad.append((x, y, z, xf, yf, zf, fl, ival, ival.err, ival.invalid))
        elif fl < ival.lo or fl > ival.hi:
            bad.append((x, y, z, xf, yf, zf, fl, ival, ival.err, ival.invalid))
    if len(bad) > 0:
        print('[FAILED] {} {}/{}'.format(name, len(bad), num_tests))
        if verbose:
            for x, y, z, xf, yf, zf, fl, ival, err, invalid in bad:
                print(' x={} ({}) y={} ({}) z={} ({}) fl={}, ival={}, err={}, invalid={}'.format(
                    str(x), str(xf), str(y), str(yf), str(z), str(zf), str(fl), str(ival), err, invalid))
    else:
        print('[PASSED] {}'.format(name))

def pow_overflow(x: ieee754.Float, y: ieee754.Float, ctx):
    ctx = x._select_context(x, y, ctx=ctx)
    result = gmpmath.compute(OP.pow, x, y, prec=ctx.p, trap_underflow=False, trap_overflow=False)
    return ieee754.Float._round_to_context(result, ctx)

def sinh_overflow(x: ieee754.Float, ctx):
    ctx = x._select_context(x, ctx=ctx)
    result = gmpmath.compute(OP.sinh, x, prec=ctx.p, trap_underflow=False, trap_overflow=False)
    return ieee754.Float._round_to_context(result, ctx)

def test_interval(num_tests=10_000, ctx=ieee754.ieee_ctx(11, 64), verbose=False):
    """Runs unit tests for the Interval type"""

    ops = [
        ("neg", ieee754.Float.neg, Interval.neg, 1),
        ("fabs", ieee754.Float.fabs, Interval.fabs, 1),
        ("add", ieee754.Float.add, Interval.add, 2),
        ("sub", ieee754.Float.sub, Interval.sub, 2),
        ("mul", ieee754.Float.mul, Interval.mul, 2),
        ("div", ieee754.Float.div, Interval.div, 2),
        ("sqrt", ieee754.Float.sqrt, Interval.sqrt, 1),
        ("cbrt", ieee754.Float.cbrt, Interval.cbrt, 1),
        ("fma", ieee754.Float.fma, Interval.fma, 3),

        ("fmin", ieee754.Float.fmin, Interval.fmin, 2),
        ("fmax", ieee754.Float.fmax, Interval.fmax, 2),
        ("fdim", ieee754.Float.fdim, Interval.fdim, 2),
        ("fmod", ieee754.Float.fmod, Interval.fmod, 2),
        ("remainder", ieee754.Float.remainder, Interval.remainder, 2),
        ("copysign", ieee754.Float.copysign, Interval.copysign, 2),

        ("rint", rint, Interval.rint, 1),
        ("round", ieee754.Float.round, Interval.round, 1),
        ("ceil", ieee754.Float.ceil, Interval.ceil, 1),
        ("floor", ieee754.Float.floor, Interval.floor, 1),
        ("trunc", ieee754.Float.trunc, Interval.trunc, 1),
        ("nearbyint", ieee754.Float.nearbyint, Interval.nearbyint, 1),

        ("exp", ieee754.Float.exp_, Interval.exp, 1),
        ("expm1", ieee754.Float.expm1, Interval.expm1, 1),
        ("exp2", ieee754.Float.exp2, Interval.exp2, 1),
        ("log", ieee754.Float.log, Interval.log, 1),
        ("log2", ieee754.Float.log2, Interval.log2, 1),
        ("log10", ieee754.Float.log10, Interval.log10, 1),
        ("log1p", ieee754.Float.log1p, Interval.log1p, 1),
        ("pow", pow_overflow, Interval.pow, 2),

        ("sin", ieee754.Float.sin, Interval.sin, 1),
        ("cos", ieee754.Float.cos, Interval.cos, 1),
        ("tan", ieee754.Float.tan, Interval.tan, 1),
        ("asin", ieee754.Float.asin, Interval.asin, 1),
        ("acos", ieee754.Float.acos, Interval.acos, 1),
        ("atan", ieee754.Float.atan, Interval.atan, 1),
        ("atan2", ieee754.Float.atan2, Interval.atan2, 2),

        ("sinh", sinh_overflow, Interval.sinh, 1),
        ("cosh", ieee754.Float.cosh, Interval.cosh, 1),
        ("tanh", ieee754.Float.tanh, Interval.tanh, 1),
        ("asinh", ieee754.Float.asinh, Interval.asinh, 1),
        ("acosh", ieee754.Float.acosh, Interval.acosh, 1),
        ("atanh", ieee754.Float.atanh, Interval.atanh, 1),

        ("erf", ieee754.Float.erf, Interval.erf, 1),
        ("erfc", ieee754.Float.erfc, Interval.erfc, 1),
    ]

    # ops = [
    #     ("pow", pow_overflow, Interval.pow, 2),
    # ]

    random.seed()
    for name, fl_fn, ival_fn, argc in ops:
        if argc == 1:
            test_unary_fn(name, fl_fn, ival_fn, num_tests, ctx, verbose)
        elif argc == 2:
            test_binary_fn(name, fl_fn, ival_fn, num_tests, ctx, verbose)
        elif argc == 3:
            test_ternary_fn(name, fl_fn, ival_fn, num_tests, ctx, verbose)
        else:
            raise ValueError('invalid argc for testing: argc={}'.format(argc))
