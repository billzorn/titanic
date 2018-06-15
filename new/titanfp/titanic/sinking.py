"""Sinking point -
A discrete approximation of real numbers with explicit significance tracking.
Implemented really badly in one file.

TODO:

in digital.py:
  - representation
  - reading in from int / mpfr / string
  - reading in from digital, clone and update
  - comparison
  - trivial bit manipulations like neg and abs
  - rounding? round only via w / p
  - output? to_mpfr? to_string? to_ieee? to_posit?

in conversion.py:
  - to / from mantissa_exp form
  - universal is_neg, etc. ?
  - extract payload

Where to put OP / RM identifiers?

in xxxmath.py:
  - numeric engine: given opcode, inputs as sink, precision,
    produce another sink with rounding status

in arithmetic:
  - round specifically to ieee / posit
  - arith wrappers that can use either backend, and do the correct rounding / special case behavior
"""

import typing
import sys
import random
import re

from .integral import bitmask
from . import conversion


def _interval_scan_away(lower, upper, n):
    """Scan for a representative with n=n whose envelope encloses lower and upper.
    Returns two things:
      True, if the interval is provably too small for the bound, else False
        (i.e. we found a representative whose envelope is totally enclosed between lower and upper).
      None, if no enclosing representative is found, else the representative.
    """
    if lower._inexact or upper._inexact:
        raise ValueError('enclose: can only scan exact numbers')
    elif lower.negative or upper.negative:
        raise ValueError('enclose: can only scan positive numbers')
    elif not lower < upper:
        raise ValueError('enclose: can only scan ordered envelope, got [{}, {}]'.format(lower, upper))

    rep = lower.trunc(n)
    if rep.is_exactly_zero():
        rep = rep.explode(sided=True, full=False)
    else:
        rep = rep.explode(sided=False, full=False)

    # This loop will only make a small number of iterations.
    # We will always hit the bottom of the interval after a short amount of time:
    # if we truncated a lot of bits off, then the interval is large and we'll hit the
    # exact value in one step. If we didn't truncate bits, the interval might be
    # small, but we'll start exactly at lower.
    # Because we detect the case where the envelope size is provable smaller than the
    # interval, we will abort after a few iterations in cases where the envelope
    # is much smaller than the interval.

    while True:
        bound_lo, bound_hi = rep.bounds()
        bottom_enclosed = bound_lo <= lower
        top_enclosed = upper <= bound_hi

        if bottom_enclosed and top_enclosed:
            # representative encloses the interval: return it
            return False, rep
        elif not (bottom_enclosed or top_enclosed):
            # interval encloses the representative: unless we were using the half envelope
            # near zero, this is proof that this n is too small
            if rep.interval_sided:
                # try the next number to see if that gives us a proof
                # TODO: sided -> sided will break everything
                rep = rep.away(const_p=False)
            else:
                return True, None
        elif bottom_enclosed:
            # (top wasn't enclosed, or we'd have hit the first case)
            # bottom of interval was good, top wasn't: move on to the next number to see what
            # happens
            rep = rep.away(const_p=False)
        else:
            # bottom of interval was no good, so we went too far.
            return False, None


def enclose(lower, upper, min_n=None):
    """Return the sink with the smallest interval that encloses lower and upper.
    Upper and lower must be exact sinks, with upper <= lower.
    TODO: auto bounds?
    TODO: other kinds of intervals?
    """
    if lower._inexact or upper._inexact:
        raise ValueError('enclose: must have exact arguments, got [{} and {}]'.format(lower, upper))
    elif lower == upper:
        return Sink(lower) if lower.n < upper.n else Sink(upper)
    elif not lower < upper:
        raise ValueError('enclose: arguments out of order, not {} < {}'.format(lower, upper))

    zero = Sink(0)
    # because upper != lower, the distance between them must be larger than the interval size
    # with this n
    min_possible_n = min(lower.n, upper.n) - 1
    if min_n is None:
        min_n = min_possible_n
    else:
        min_n = max(min_possible_n, min_n)

    if lower < zero and upper > zero:
        # binsearch around zero
        offset = 1
        n_lo = n_hi = min_n
        bound_lo, bound_hi = zero.trunc(n_hi).explode(sided=False, full=False).bounds()
        # first expsearch for n_hi
        while lower < bound_lo or bound_hi < upper:
            n_lo = n_hi
            n_hi = n_hi + offset
            offset <<= 1
            bound_lo, bound_hi = zero.trunc(n_hi).explode(sided=False, full=False).bounds()
        # final condition: n_hi, bound_lo, bound_hi are all safe
        while n_lo + 1 < n_hi:
            n_mid = n_lo + ((n_hi - n_lo) // 2)
            bound_lo, bound_hi = zero.trunc(n_mid).explode(sided=False, full=False).bounds()
            if lower < bound_lo or bound_hi < upper:
                # bound is unsafe, update n_lo
                n_lo = n_mid
            else:
                # bound is safe, update n_hi
                n_hi = n_mid
        # final conditions: n_lo + 1 = n_hi, n_lo doesn't work, n_hi works
        # OR, we never entered the loop, and n_lo = n_hi = min_n
        return zero.trunc(n_hi).explode(sided=False, full=False)

    else:
        # First, reorder based on magnitude, as we can only trunc towards zero.
        if lower.negative:
            tmp = -lower
            lower = -upper
            upper = tmp
            negative = True
        else:
            negative = False

        # Binsearch for the largest interval that doesn't work.
        # We know we've found it when we can demonstrate that the span
        # of this interval is too small, but the demonstration fails for the next size up.
        offset = 1
        n_lo = n_hi = min_n
        too_small, enclosing_rep = _interval_scan_away(lower, upper, n_hi)
        # first expsearch for n_hi
        while too_small:
            n_lo = n_hi
            n_hi = n_hi + offset
            offset <<= 1
            too_small, enclosing_rep = _interval_scan_away(lower, upper, n_hi)
        # final condition: n_hi is not provably too small
        while n_lo + 1 < n_hi:
            n_mid = n_lo + ((n_hi - n_lo) // 2)
            too_small, enclosing_rep = _interval_scan_away(lower, upper, n_mid)
            if too_small:
                # provably too small, update n_lo
                n_lo = n_mid
            else:
                # not provable: update n_hi
                n_hi = n_mid
        # final conditions: n_lo + 1 = n_hi, n_lo is provably too small, n_hi has no such proof
        # OR, we never entered the loops, and n_lo = n_hi = min_n

        # We now perform a linear search, starting from n_lo, until we find the smallest n
        # that can produce a representative. This should not take very long, as we are doubling
        # the size of the envelope each time we increment n.
        # TODO: We could save a few cycles by refusing to actually test n_lo if it is the same as n_hi.
        n = n_lo
        while True:
            too_small, enclosing_rep = _interval_scan_away(lower, upper, n)
            if enclosing_rep is None:
                n += 1
            else:
                # remember to correct the sign
                return Sink(enclosing_rep, negative=negative)


class Sink(object):

    # for sinks with a real value, the value is exactly  (sign) * _c * 2**_exp
    _c : int = None # unsigned significand
    _exp : int = None # exponent

    # sign is stored separately, as is information about infiniteness or NaN
    _negative : bool = None # sign bit
    _isinf : bool = None # is this value infinite?
    _isnan : bool = None # is this value NaN?

    # _m and _exp are not directly visible; we expose them with attributes

    @property
    def m(self):
        """Signed integer significand.
        The real value is m * 2**exp
        """
        if self._negative:
            return -self._c
        else:
            return self._c

    @property
    def exp(self):
        """Exponent."""
        return self._exp

    # we also present 4 views for the primary 'Titanic' properties

    @property
    def e(self):
        """IEEE 754 style exponent.
        If the significand is interpreted as a binary-point number x between 1 and 2,
        i.e. 1.10011100 etc. then the real value is x * 2**e.
        """
        return (self._exp - 1) + self._c.bit_length()

    @property
    def n(self):
        """The 'sticky bit' or the binary place where digits are no longer significant.
        I.e. -1 for an integer inexact beyond the binary point. Always equal to exp - 1.
        """
        return self._exp - 1

    @property
    def p(self):
        """The precision of the significand.
        Always equal to the number of bits in c; 0 for any zero.
        """
        return self._c.bit_length()

    @property
    def c(self):
        """Unsigned integer significand."""
        return self._c

    # views of basic semantic flags

    @property
    def negative(self):
        """The sign bit - is this value negative?"""
        return self._negative

    @property
    def isinf(self):
        """Is this value infinite?"""
        return self._isinf

    @property
    def isnan(self):
        """Is this value NaN?"""
        return self._isnan


    # rounding envelopes and inexactness
    _inexact : bool = None # approximate bit
    _interval_full : bool = None # envelope interval size
    _interval_sided : bool = None # envelope interval position
    _interval_open_top : bool = None # is the top bound exclusive?
    _interval_open_bottom : bool = None # ditto for the bottom bound
    _rc : int = None # as MPFR result code. 0 if value is exact, -1 if rounded up, 1 if rounded down.

    # views for interval properties

    @property
    def inexact(self):
        """Is this value inexact?"""
        return self._inexact

    @property
    def interval_full(self):
        """Does the rounding envelope for this number extend a full ulp
        on each side? (if false, it is a half ulp)
        """
        return self._interval_full

    @property
    def interval_sided(self):
        """Does the rounding envelope only extend away from zero?
        (if False, it is symmetric on both sides)
        """
        return self._interval_sided

    @property
    def interval_open_top(self):
        """Is the top of the rounding envelope exclusive?
        (if False, it is inclusive, or closed)
        """
        return self._interval_open_top

    @property
    def interval_open_bottom(self):
        """Is the bottom of the rounding envelope exclusive?
        (if False, it is inclusive, or closed)
        """
        return self._interval_open_bottom

    @property
    def rc(self):
        """Result code. -1 if this value was rounded up, 1 if it was rounded down,
        0 if the value was computed exactly or rounding direction is unknown.
        """
        return self._rc


    # other useful properties

    def is_exactly_zero(self):
        return self._c == 0 and not self._inexact

    def is_zero(self):
        return self._c == 0

    def is_integer(self):
        return self._exp >= 0 or self._c & bitmask(-self._exp) == 0

    def is_identical_to(self, x):
        return (
            self._c == x._c
            and self._exp == x._exp
            and self._negative == x._negative
            and self._isinf == x._isinf
            and self._isnan == x._isnan
            and self._inexact == x._inexact
            and self._interval_full == x._interval_full
            and self._interval_sided == x._interval_sided
            and self._interval_open_top == x._interval_open_top
            and self._interval_open_bottom == x._interval_open_bottom
            and self._rc == x._rc
        )

    def __init__(self,
                 # The base value of the sink, either as a sink to copy
                 # or a string / float / mpfr to parse.
                 base=None,
                 # value information about the sink to construct
                 m=None,
                 exp=None,
                 # either m must be specified, or c and negative must be specified
                 c=None,
                 # negative can be specified alone to change the sign
                 negative=None,
                 # inf and nan can be set independently of other properties of the
                 # number, though having both set at once is not well defined
                 inf=None,
                 nan=None,
                 # inexactness information can be specified or modified independently
                 inexact=None,
                 full=None,
                 sided=None,
                 open_top=None,
                 open_bottom=None,
                 rc=None,
                 # rounding properties; ignored unless parsing a string
                 max_p=None,
                 min_n=None,
                 rm=conversion.ROUND_NEAREST_EVEN
    ):
        """Create a new sinking point number. The value can be specified in 3 ways:

        If base is None, then the new number must have its value specified by exp
        and either m, or c and negative. Note that since integer 0 in Python does
        not have a sign, a signed zero must be specified with c and negative.

        If base is an existing Sink, then that number is copied, and its fields can
        be updated individually.

        If base is a numeric type or a string, then that number is converted to a sink
        with the closest possible value, as per the rounding specification. In practice,
        rounding will only occur for strings. If the specified rounding is impossible
        (i.e. rm is None, or both max_p and min_n are unspecified for a value such as
        Pi with no finite representation) then an exception will be raised.
        """

        # raw, non-converting forms
        if base is None or isinstance(base, Sink):

            # create from mantissa / exponent form
            if base is None:
                if not ((m is not None and (c is None and negative is None))
                        or (m is None and (c is not None and negative is not None))):
                    raise ValueError('must specify either m, or c and negative')
                elif inf and nan:
                    raise ValueError('number cannot be simultaneously inf and nan')

                if m is not None:
                    self._c = abs(m)
                    self._negative = (m < 0)
                else:
                    self._c = c
                    self._negative = negative

                self._exp = exp

                self._isinf = bool(inf)
                self._isnan = bool(nan)

                self._inexact = bool(inexact)
                self._interval_full = bool(full)
                self._interval_sided = bool(sided)
                self._interval_open_top = bool(open_top)
                self._interval_open_bottom = bool(open_bottom)

                if rc is None:
                    self._rc = 0
                else:
                    self._rc = rc

            # copy from existing sink
            else:
                if m is not None and (c is not None or negative is not None):
                    raise ValueError('cannot specify c or negative if m is specified')

                if m is not None:
                    self._c = abs(m)
                    self._negative = (m < 0)
                else:
                    self._c = c if c is not None else base.c
                    self._negative = negative if negative is not None else base.negative

                self._exp = exp if exp is not None else base.exp

                self._isinf = inf if inf is not None else base.isinf
                self._isnan = nan if nan is not None else base.isnan

                if self.isnan and self.isinf:
                    raise ValueError('cannot update number to simultaneously be inf and nan')

                self._inexact = inexact if inexact is not None else base.inexact
                self._interval_full = full if full is not None else base.interval_full
                self._interval_sided = sided if sided is not None else base.interval_sided
                self._interval_open_top = open_top if open_top is not None else base.interval_open_top
                self._interval_open_bottom = open_bottom if open_bottom is not None else base.interval_open_bottom
                self._rc = rc if rc is not None else base.rc

        # convert another representation into sinking point
        else:
            if not (m is None and exp is None and c is None and negative is None and inf is None and nan is None):
                raise ValueError('cannot specify numeric properties when converting another numeric type')

            if isinstance(base, str):
                # TODO unimplemented
                base = float(base)

            # TODO does not support inf and nan
            negative, c, exp = conversion.numeric_to_signed_mantissa_exp(base)

            self._c = c
            self._negative = negative
            self._exp = exp
            self._isinf = False
            self._isnan = False

            # TODO conflict with rounding
            self._inexact = bool(inexact)
            self._interval_full = bool(full)
            self._interval_sided = bool(sided)
            self._interval_open_top = bool(open_top)
            self._interval_open_bottom = bool(open_bottom)

            # round to specified precision


    def __repr__(self):
        return 'Sink({}, c={}, exp={},  negative={}, inexact={}, full={}, sided={}, rc={})'.format(
            self.to_mpfr(), self.c, self.exp, self.negative, self.inexact, self.interval_full, self.interval_sided, self.rc,
        )

    def __str__(self):
        """yah"""
        if self.c == 0:
            sgn = '-' if self.negative else ''
            if self._inexact:
                return '{}0~@{:d}'.format(sgn, self.n)
            else:
                #print(repr(self))
                return '{}0'.format(sgn)
        else:
            rep = re.search(r"'(.*)'", repr(self.to_mpfr())).group(1).split('e')
            s = rep[0]
            sexp = ''
            if len(rep) > 1:
                sexp = 'e' + 'e'.join(rep[1:])
            return '{}{}{}'.format(s, '~' if self._inexact else '', sexp)
            # return '{}{}'.format(rep, '~@{:d}'.format(self.n) if self._inexact else '')


    def round_m(self, max_p, min_n=None):
        #TODO: make this better
        return self.widen(max_p=max_p, min_n=min_n)

    # core envelope operations


    # Adjacent interval logic.
    # If const_p is True, then preserve the value of p (this is the behavior of IEEE754 FP).
    # Otherwise, preserve n - this ensures intervals have the same size, as for fixed point.
    # If strict is True, then always preserve interval properties - this may produce a disconnected interval
    # for half intervals. Otherwise, sided half intervals will produce (connected) unsided half intervals,
    # and unsided intervals will flow through sided intervals around zero.

    # TODO: toward for sided half intervals produces a (still disconnected) unsided half interval.

    # TODO: the low-level semantics of this are not necessarily reasonable nor important


    def away(self, const_p = False, strict = False):
        """The sink with the next greatest magnitude at this precision, away from 0.
        Preserves sign and exactness. Meaningless for non-sided zero.
        """
        if self.is_zero() and (not self.interval_sided):
            raise ValueError('away: cannot determine which direction to go from {}'.format(repr(self)))

        next_c = self.c + 1
        next_exp = self.exp

        if next_c.bit_length() > self.p:
            if const_p and next_c > 1:
                # normalize precision, if we want to keep it constant
                # only possible if we didn't start from 0
                # TODO this definition of constant precision is broken, use IEEE 754 max_p / min_n
                next_c >>= 1
                next_exp += 1

        if strict:
            sided = self.interval_sided
        else:
            if next_c == 1:
                sided = False
            elif not self.interval_full:
                sided = False
            else:
                sided = self.interval_sided

        return Sink(self, c=next_c, exp=next_exp, sided=sided)


    def toward(self, const_p = False, strict = False):
        """The sink with the next smallest magnitude at this precision, toward 0.
        Preserves sign and exactness. Meaningless for any zero.
        """
        if self.is_zero():
            raise ValueError('toward: {} is already 0'.format(repr(self)))

        prev_c = self.c - 1
        prev_exp = self.exp

        if prev_c.bit_length() < self.c.bit_length():
            if const_p and prev_c > 0:
                # normalize precision, if we want to keep it constant
                # only possible if we didn't actually reach 0
                # TODO this definition of constant precision is broken, use IEEE 754 max_p / min_n
                prev_c <<= 1
                prev_exp -= 1

        if strict:
            sided = self.interval_sided
        else:
            if prev_c == 0:
                sided = True
            elif not self.interval_full:
                sided = False
            else:
                sided = self.interval_sided

        return Sink(self, c=prev_c, exp=prev_exp, sided=sided)


    def above(self, const_p = False, strict = False):
        """The sink with the next largest value, toward positive infinity.
        """
        if self.is_zero():
            if self.interval_sided:
                if self.negative:
                    return -self
                else:
                    return self.away(const_p=const_p, strict=strict)
            else:
                if strict:
                    sided = self.interval_sided
                else:
                    sided = False
                return Sink(self, c=1, negative=False, sided=sided)
        elif self.negative:
            return self.toward(const_p=const_p, strict=strict)
        else:
            return self.away(const_p=const_p, strict=strict)


    def below(self, const_p = False, strict = False):
        """The sink with the next smallest value, toward negative infinity.
        """
        if self.is_zero():
            if self.interval_sided:
                if self.negative:
                    return self.away(const_p=const_p, strict=strict)
                else:
                    return -self
            else:
                if strict:
                    sided = self.interval_sided
                else:
                    sided = False
                return Sink(self, c=1, negative=True, sided=sided)
        elif self.negative:
            return self.away(const_p=const_p, strict=strict)
        else:
            return self.toward(const_p=const_p, strict=strict)


    # Interval representatives and bounds.
    # An interval's representative is the exact value used for arithmetic in traditional
    # IEEE 754-like systems. An interval's bounds are [inclusive] limits on the values the interval
    # can represent. For half intervals, they will have one more bit of precision than the
    # interval's representative.

    # TODO: bounds are always inclusive; this could be tracked, for example to actually do the right
    # thing with <> and rounding modes.


    def collapse(self, center=False):
        """Collapse an interval down to a representative point.
        For sided intervals, can return the "bottom" of the interval, or its true center, which requires
        1-2 bits more precision.
        """
        if center and self.interval_sided and self._inexact:
            extra_bits = 1 if self.interval_full else 2
            return Sink(self.narrow(n=self.n - extra_bits), inexact=False, sided=False).away()
        else:
            return Sink(self, inexact=False)


    def explode(self, sided=None, full=None):
        """Explode a representative point to an enclosing interval.
        If provided, sided and full replace the corresponding properties of the original interval.
        It is invalid to explode a larger interval to a smaller one, i.e. full to half or
        unsided to sided.
        """
        if self._inexact:
            if sided and (not self.interval_sided):
                raise ValueError('explode: cannot shrink unsided interval {} to sided'.format(repr(self)))
            elif full and (not self.interval_full):
                raise ValueError('explode: cannot shrink full interval {} to half'.format(repr(self)))

        sided = self.interval_sided if sided is None else sided
        full = self.interval_full if full is None else full
        return Sink(self, inexact=True, sided=sided, full=full)


    def bounds(self):
        """Upper and lower bounds on the value of this number.
        Intervals are inclusive.
        """
        if self._inexact:
            if self.interval_full:
                base = self
            else:
                base = self.narrow(n=self.n - 1)

            if self.interval_sided:
                if self.negative:
                    return base.away().collapse(), self.collapse()
                else:
                    return self.collapse(), base.away().collapse()
            else:
                return base.below().collapse(), base.above().collapse()
        else:
            return Sink(self), Sink(self)


    def trunc(self, n):
        """Round this number towards 0, throwing away the low bits, or append zeros
        onto the end, to provide a lower bound on its absolute value at any n.
        """
        if self._inexact:
            # TODO
            raise ValueError('trunc: unsupported: inexact value {}'.format(repr(self)))

        if self.n == n:
            return Sink(self)
        else:
            if self.n < n:
                # get rid of bits
                offset = n - self.n
                c = self.c >> offset
                exp = self.exp + offset
            else:
                # add bits
                offset = self.n - n
                c = self.c << offset
                exp = self.exp - offset
            return Sink(self, c=c, exp=exp)


    def split(self, n=None, rm=0):
        """Split a number into an exact part and an uncertainty bound.
        If we produce split(A, n) -> A', E, then we know:
          - A' is exact
          - E is zero
          - lsb(A') == lsb(E) == max(n, lsb(A)) if A is inexact
          - lsb(A') == lsb(E) == n if A is exact
        TODO: is this correct????
        """
        if n is None:
            n = self.n
        offset = n - self.n

        if offset <= 0:
            if offset == 0 or self.inexact:
                return (Sink(self, inexact=False), Sink(self, c=0))
            else:
                return (Sink(self, c=self.c << -offset, exp=n+1), Sink(self, c=0, exp=n+1))
        else:
            lost_bits = self.c & bitmask(offset)
            left_bits = self.c >> offset
            low_bits = lost_bits & bitmask(offset - 1)
            half_bit = lost_bits >> (offset - 1)

            inexact = self._inexact or lost_bits != 0
            if left_bits == 0 and lost_bits != 0:
                sided = True
            else:
                sided = self.interval_sided

            rounded = Sink(self, c=left_bits, exp=n+1, inexact=False, sided=sided)
            # in all cases we copy the sign onto epsilon... is that right?
            epsilon = Sink(self, c=0, exp=n+1, inexact=inexact, sided=sided)

            # TODO use sane RM
            if half_bit == 1:
                # Note that if we're rounding an inexact number, then the new tight 1-ulp envelope
                # of the result will not contain the entire envelope of the input.
                if low_bits == 0:
                    # Exactly half way between, regardless of exactness.
                    # Use rounding mode to decide.
                    if rm == 0:
                        # round to even if rm is zero
                        if left_bits & bitmask(1) == 1:
                            return rounded.away(const_p=False), epsilon
                        else:
                            return rounded, epsilon
                    elif rm > 0:
                        # round away from zero if rm is positive
                        return rounded.away(const_p=False), epsilon
                    else:
                        # else, round toward zero if rm is negative
                        return rounded, epsilon
                else:
                    return rounded.away(const_p=False), epsilon
            else:
                return rounded, epsilon


    def widen(self, min_n = None, max_p = None):
        """Round this number, using split, so that n is >= min_n and p <= max_p.
        By default, preserve n and p, returning this number unchanged.
        """
        if min_n is None:
            n = self.n
        else:
            n = min_n

        if max_p is not None:
            n = max(n, self.e - max_p)

        rounded, epsilon = self.split(n)

        if max_p is not None and rounded.p > max_p:
            # If we rounded up and carried, we might have increased p by one.
            # Split again to compensate; this should produce an epsilon of zero.
            rounded, epsilon_correction = rounded.split(n + 1)
            if not epsilon_correction.is_exactly_zero():
                epsilon = epsilon_correction
                raise ValueError('widen: unreachable')

        return Sink(rounded, inexact=epsilon.inexact)


    def narrow(self, n=None, p=None):
        """Force this number into a representation with either n or p.
        By default, preserve n and p, returning this number unchanged.
        Note that this may produce a smaller envelope that does not contain
        the input value.
        """
        if n is p is None:
            return Sink(self)
        elif n is None:
            if self.c == 0:
                # TODO what should be done here?
                # specifying precision is meaningless for zero
                n = self.n
            else:
                n = self.e - p
        elif p is None:
            # use n as provided
            pass
        else:
            raise ValueError('narrow: can only specify one of n or p, got n={}, p={}'
                             .format(repr(n), repr(p)))

        rounded, epsilon = self.split(n)

        # There are two possibilities:
        # Either we are trying to narrow the envelope, i.e. increase precision,
        # and this split was a no-op;
        # Or we are actually trying to widen the envelope, i.e. decrease precision,
        # and this split may have rounded up, giving us more precision than we want.

        if rounded.n > n:
            # split was unable to provide a small enough n, so we have to force one
            rounded = Sink(rounded, c=rounded.c << (rounded.n - n), exp=n+1)
        elif p is not None and rounded.p > p:
            # as for widening, round again to compensate
            rounded, epsilon_correction = rounded.split(n + 1)
            if not epsilon_correction.is_exactly_zero():
                epsilon = epsilon_correction
                raise ValueError('narrow: unreachable')

        return Sink(rounded, inexact=epsilon.inexact)


    def ieee_754(self, w, p):
        emax = (1 << (w - 1)) - 1
        emin = 1 - emax
        max_p = p
        min_n = emin - p

        if self.c == 0:
            return self.narrow(n=min_n)
        elif self.n <= min_n or self.p <= max_p:
            return self.widen(min_n=min_n, max_p=max_p)
        else:
            extra_bits = p - self.p
            return self.narrow(n=max(min_n, self.n - extra_bits))


    def to_mpfr(self):
        if self.negative:
            return conversion.mpfr_from_mantissa_exp(-self.c, self.n + 1)
        else:
            return conversion.mpfr_from_mantissa_exp(self.c, self.n + 1)


    def to_float(self, ftype=float):
        data = conversion.fdata(ftype)
        w = data['w']
        p = data['p']

        rounded = self.ieee_754(w, p)

        if rounded.negative:
            return conversion.float_from_mantissa_exp(-rounded.c, rounded.n + 1, ftype=ftype)
        else:
            return conversion.float_from_mantissa_exp(rounded.c, rounded.n + 1, ftype=ftype)


    def to_math(self):
        # TODO assumes exactness
        if self.is_zero():
            return '0'
        elif self.is_integer() and self.exp < 0:
            return str(self.m >> -self.exp)
        elif self.is_integer() and self.exp < 32: # TODO some reasonable threshold
            return str(self.m << self.exp)
        else:
            return '{:d} * 2^{:d}'.format(self.m, self.exp)


    # core arith and comparison


    def __neg__(self):
        return Sink(self, negative=not self.negative)


    def __abs__(self):
        return Sink(self, negative=False)


    def compareto(self, x, strict=True):
        """Compare to another number.
        Returns two different things: the ordering, and the sharpness.
        For a.compareto(b), the ordering is:
         -1 iff a < b
          0 iff a = b
          1 iff a > b
        And the sharpness is:
          True iff the intervals do not overlap, or a and b are the same point
          False iff the intervals overlap at a single point (i.e. they are touching)
          None iff the intervals overlap for a region larger than a single point
        Note that two identical points have a sharpness of False, rather than None.
        """
        lower, upper = self.bounds()
        xlower, xupper = x.bounds()

        # normalize to smallest n
        n = min(upper.n, lower.n, xupper.n, xlower.n)
        lower = lower.narrow(n=n)
        upper = upper.narrow(n=n)
        xlower = xlower.narrow(n=n)
        xupper = xupper.narrow(n=n)

        # convert to ordinals
        lower_ord = -lower.c if lower.negative else lower.c
        upper_ord = -upper.c if upper.negative else upper.c
        xlower_ord = -xlower.c if xlower.negative else xlower.c
        xupper_ord = -xupper.c if xupper.negative else xupper.c

        # integer comparison
        if not (lower_ord <= upper_ord and xlower_ord <= xupper_ord):
            # TODO: assertion
            print(lower_ord, upper_ord, xlower_ord, xupper_ord)
            raise ValueError('compareto: unreachable')
        elif lower_ord == upper_ord == xlower_ord == xupper_ord:
            # a == b
            order = 0
            sharp = True
        elif upper_ord <= xlower_ord:
            # a <= b
            order = -1
            sharp = upper_ord != xlower_ord
        elif xupper_ord < lower_ord:
            # b <= a
            order = 1
            sharp = xupper_ord != lower_ord
        else:
            # overlap: compare representatives
            # TODO: center here? it makes comparisons fair...
            center = False
            rep = self.collapse(center=center)
            xrep = x.collapse(center=center)

            n = min(rep.n, xrep.n)
            rep = rep.narrow(n=n)
            xrep = xrep.narrow(n=n)
            rep_ord = -rep.c if rep.negative else rep.c
            xrep_ord = -xrep.c if xrep.negative else xrep.c

            if rep == xrep:
                # a == b
                order = 0
            elif rep < xrep:
                # a < b
                order = -1
            else:
                # b < a
                order = 1

            sharp = None

        if strict and sharp is None:
            # TODO: this will print warnings, but has no other teeth, and is otherwise unused
            # in inline comparisons.
            print('WARNING: compared overlapping intervals {} and {}'.format(self, x))

        return order, sharp


    def __lt__(self, x):
        order, sharp = self.compareto(x)
        if sharp is False:
            # TODO: fangs
            print('WARNING: {} < {} is not known to be sharp'.format(self, x))
        return order < 0

    def __le__(self, x):
        order, sharp = self.compareto(x)
        return order <= 0

    def __eq__(self, x):
        order, sharp = self.compareto(x)
        return order == 0

    def __ne__(self, x):
        order, sharp = self.compareto(x)
        return order != 0

    def __ge__(self, x):
        order, sharp = self.compareto(x)
        return 0 <= order

    def __gt__(self, x):
        order, sharp = self.compareto(x)
        if sharp is False:
            # TODO: fangs
            print('WARNING: {} > {} is not known to be sharp'.format(self, x))
        return 0 < order


    #TODO: arith
    #TODO: precision explodes with unnecessary trailing zeros, which is probably bad...

    def __add__(self, x):
        """Add this sink to another sink x, exactly. Fails if either is inexact."""
        if self.inexact or x.inexact:
            raise ValueError('add: can only add exact sinks, got {} + {}'.format(repr(self), repr(x)))
        n = min(self.n, x.n)
        c_norm = self.c << (self.n - n)
        xc_norm = x.c << (x.n - n)
        sign = -1 if self.negative else 1
        xsign = -1 if x.negative else 1
        signed_c = (sign * c_norm) + (xsign * xc_norm)
        if signed_c >= 0:
            c = signed_c
            negative = False
        else:
            c = -signed_c
            negative = True

        #TODO: inf and nan
        #TODO: sign of negative 0
        #TODO: envelope properties
        return Sink(self, c=c, exp=n+1, negative=negative, sided=False)


    def __sub__(self, x):
        """Alias of self + (-x)"""
        return self + (-x)


    def __mul__(self, x):
        """Multiply this sink by another sink x, exactly. Fails if either is inexact."""
        if self.inexact or x.inexact:
            raise ValueError('mul: can only multiply exact sinks, got {} * {}'.format(repr(self), repr(x)))
        n = self.n + x.n + 1 # equivalent to (self.n + 1) + (x.n + 1) - 1
        c = self.c * x.c
        # TODO assert
        if self.negative is None or x.negative is None:
            raise ValueError('mul {} * {}: negative is None'.format(repr(self), repr(x)))
        negative = self.negative != x.negative
        #TODO: inf and nan
        #TODO: envelope properties
        return Sink(self, c=c, exp=n+1, negative=negative, sided=False)
