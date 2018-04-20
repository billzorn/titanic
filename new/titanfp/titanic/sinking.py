"""Sinking point -
A discrete approximation of real numbers with explicit significance tracking.
Implemented really badly in one file.
"""


import typing
import sys
import random
import re

# ideally these would not be here...
# for example, put some sort of fdata() function in conversion to get w and p...
import numpy as np
import gmpy2 as gmp

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
    elif lower._negative or upper._negative:
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
            if rep._sided_interval:
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
        return Sink(lower) if lower._n < upper._n else Sink(upper)
    elif not lower < upper:
        raise ValueError('enclose: arguments out of order, not {} < {}'.format(lower, upper))

    zero = Sink(0)
    # because upper != lower, the distance between them must be larger than the interval size
    # with this n
    min_possible_n = min(lower._n, upper._n) - 1
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
        if lower._negative:
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
    _e : int = None # exponent
    _n : int = None # "sticky bit" or lsb
    _p : int = None # precision: e - n
    _c : int = None # significand
    _negative : bool = None # sign bit
    _inexact : bool = None # approximate bit
    _full_interval : bool = None # envelope interval size
    _sided_interval : bool = None # envelope interval position
    _isinf : bool = None # is the value infinite?
    _isnan : bool = None # is this value NaN?


    def _valid(self) -> bool:
        return (
            (self._e >= self._n) and
            (self._p == self._e - self._n) and
            (self._c.bit_length() == self._p) and
            (self._c >= 0) and
            # no support for nonfinite yet
            (not (self._isinf or self._isnan))
        )


    def __init__(self, x=None, e=None, n=None, p=None, c=None,
                 negative=None, inexact=None, sided_interval=None, full_interval=None,
                 isinf=None, isnan=None,
                 max_p=None, min_n=None) -> None:
        """Create a new Sink.
        If an existing Sink is provided, then the fields can be specified individually
        as arguments to the constructor.
        If a new sink is being created, then most fields will be ignored, except n for
        the lsb of 0 values and p for the precision of mpfrs.
        Note that __init__ is currently recursive, to handle some cases of 0 and
        round-on-init with max_p and min_n.
        TODO TODO TODO
        """

        #print('sink {} with inexact={}'.format(repr(x), inexact))

        # if given another sink, clone and update
        if isinstance(x, Sink):
            # might have to think about this more carefully...
            self._e = x._e if e is None else int(e)
            self._n = x._n if n is None else int(n)
            self._p = x._p if p is None else int(p)
            self._c = x._c if c is None else int(c)
            self._negative = x._negative if negative is None else bool(negative)
            self._inexact = x._inexact if inexact is None else bool(inexact)
            self._sided_interval = x._sided_interval if sided_interval is None else bool(sided_interval)
            self._full_interval = x._full_interval if full_interval is None else bool(full_interval)
            self._isinf = x._isinf if isinf is None else bool(isinf)
            self._isnan = x._isnan if isnan is None else bool(isnan)

        # By default, produce "zero".
        # Note that this throws away the sign of the zero, and substitutes the provided sign
        # and interval specification.
        # TODO
        elif x is None:
            if n is None:
                raise ValueError('zero must specify n')
            else:
                self._e = self._n = int(n)
                self._p = self._c = 0
                self._inexact = bool(inexact)
                self._negative = bool(negative)
                self._sided_interval = bool(sided_interval)
                self._full_interval = bool(full_interval)
                self._isinf = self._isnan = False

        # integers are exact and have n=-1
        elif isinstance(x, int):
            self._c = abs(x)
            self._p = self._c.bit_length()
            self._n = -1
            self._e = self._n + self._p
            self._negative = x < 0
            self._inexact = False
            self._sided_interval = False
            self._full_interval = False
            self._isinf = self._isnan = False

        # otherwise convert from mpfr
        # TODO: get incoming precision right (custom parser)
        else:
            # guess precision for
            if p is max_p is None:
                prec = _DEFAULT_PREC
            elif p is None:
                prec = max_p
            else:
                prec = p

            # pi hack
            if isinstance(x, str) and x.strip().lower() == 'pi':
                with gmp.context(precision=prec) as gmpctx:
                    x = gmp.const_pi()
                    inexact = True

            if not isinstance(x, mpfr_t):
                with gmp.context(precision=prec) as gmpctx:
                    x = gmp.mpfr(x)

            # we reread precision from the mpfr
            m, exp = conversion.mpfr_to_mantissa_exp(x)
            if m == 0:
                # negative is disregarded in this case, only inexact is passed through
                #print('a zero')
                self.__init__(x=0, n=x.precision, inexact=inexact)
            else:
                self._c = abs(int(m))
                self._p = m.bit_length()
                self._n = int(exp) - 1
                self._e = self._n + self._p
                self._inexact = inexact
                # all intervals are half / unsided due to RNE
                self._full_interval = self._sided_interval = False
                self._isinf = self._isnan = False

                if negative is None:
                    self._negative = m < 0
                else:
                    if m < 0:
                        raise ValueError('negative magnitude')
                    self._negative = negative

        if not max_p is min_n is None:
            # TODO: not sound to round!
            self.__init__(self.widen(min_n=min_n, max_p=max_p))

        assert self._valid()


    def __repr__(self):
        return 'Sink({}, e={}, n={}, p={}, c={}, negative={}, inexact={}, sided_interval={}, full_interval={})'.format(
            self.to_mpfr(), self._e, self._n, self._p, self._c, self._negative, self._inexact, self._sided_interval, self._full_interval
        )

    def __str__(self):
        """yah"""
        if self._c == 0:
            sgn = '-' if self._negative else ''
            if self._inexact:
                return '{}0~@{:d}'.format(sgn, self._n)
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
            # return '{}{}'.format(rep, '~@{:d}'.format(self._n) if self._inexact else '')


    def details(self):
        try:
            mpfr_val = self.to_mpfr()
        except Exception as exc:
            mpfr_val = exc
        try:
            f64_val = self.to_float(np.float64)
        except Exception as exc:
            f64_val = exc
        try:
            f32_val = self.to_float(np.float32)
        except Exception as exc:
            f32_val = exc
        try:
            f16_val = self.to_float(np.float16)
        except Exception as exc:
            f16_val = exc

        print ('Sinking point number:\n  e={}\n  n={}\n  p={}\n  c={}\n  negative={}\n  inexact={}\n  sided={}\n  full={}\n  isinf={}\n  isnan={}\n  valid? {}'
               .format(self._e, self._n, self._p, self._c, self._negative, self._inexact, self._sided_interval, self._full_interval, self._isinf, self._isnan, self._valid()) +
               '\n    as mpfr: {}\n    as np.float64: {}\n    as np.float32: {}\n    as np.float16: {}'
               .format(repr(mpfr_val), repr(f64_val), repr(f32_val), repr(f16_val)))


    # properties


    def is_exactly_zero(self) -> bool:
        """Really there are multiple kinds of 0:
          - 'Exactly' 0, as written: self._inexact == False and self._sided_interval == False
          - 0 or infinitely close to 0, from either side: lim(n) as n -> 0: self._inexact == False and self._sided_interval == True
          - finitely close to 0, from either side: lim(n) as n -> small: self._inexact == True and self._sided_interval == True
          - finitely close to zero from some side, side unknown: self._inexact == True and self._sided_interval == False
        This just checks for either of the first two kinds, that are infinitely close to 0.
        """
        return self._c == 0 and (not self._inexact)


    # core envelope operations


    # Adjacent interval logic.
    # If const_p is True, then preserve the value of p (this is the behavior of IEEE754 FP).
    # Otherwise, preserve n - this ensures intervals have the same size, as for fixed point.
    # If strict is True, then always preserve interval properties - this may produce a disconnected interval
    # for half intervals. Otherwise, sided half intervals will produce (connected) unsided half intervals,
    # and unsided intervals will flow through sided intervals around zero.

    # TODO: toward for sided half intervals produces a (still disconnected) unsided half interval.


    def away(self, const_p = False, strict = False):
        """The sink with the next greatest magnitude at this precision, away from 0.
        Preserves sign and exactness. Meaningless for non-sided zero.
        """
        if self._c == 0 and (not self._sided_interval):
            raise ValueError('away: cannot determine which direction to go from {}'.format(repr(self)))

        next_e = self._e
        next_c = self._c + 1
        next_n = self._n
        next_p = self._p

        if next_c.bit_length() > self._c.bit_length():
            # adjust e if we carried
            next_e += 1
            if const_p and next_c > 1:
                # normalize precision, if we want to keep it constant
                # only possible if we didn't start from 0
                next_c >>= 1
                next_n += 1
            else:
                next_p += 1

        if strict:
            sided = self._sided_interval
        else:
            if next_c == 1:
                sided = False
            elif not self._full_interval:
                sided = False
            else:
                sided = self._sided_interval

        return Sink(self, e=next_e, n=next_n, p=next_p, c=next_c, sided_interval=sided)


    def toward(self, const_p = False, strict = False):
        """The sink with the next smallest magnitude at this precision, toward 0.
        Preserves sign and exactness. Meaningless for any zero.
        """
        if self._c == 0:
            raise ValueError('toward: {} is already 0'.format(repr(self)))

        prev_e = self._e
        prev_c = self._c - 1
        prev_n = self._n
        prev_p = self._p

        if prev_c.bit_length() < self._c.bit_length():
            # adjust e if we borrowed
            prev_e -= 1
            if const_p and prev_c > 0:
                # normalize precision, if we want to keep it constant
                # only possible if we didn't actually reach 0
                prev_c <<= 1
                prev_n -= 1
            else:
                prev_p -= 1

        if strict:
            sided = self._sided_interval
        else:
            if prev_c == 0:
                sided = True
            elif not self._full_interval:
                sided = False
            else:
                sided = self._sided_interval

        return Sink(self, e=prev_e, n=prev_n, p=prev_p, c=prev_c)


    def above(self, const_p = False, strict = False):
        """The sink with the next largest value, toward positive infinity.
        """
        if self._c == 0:
            if self._sided_interval:
                if self._negative:
                    return -self
                else:
                    return self.away(const_p=const_p, strict=strict)
            else:
                if strict:
                    sided = self._sided_interval
                else:
                    sided = False
                return Sink(self, e=self._n+1, p=1, c=1, negative=False, sided_interval=sided)
        elif self._negative:
            return self.toward(const_p=const_p, strict=strict)
        else:
            return self.away(const_p=const_p, strict=strict)


    def below(self, const_p = False, strict = False):
        """The sink with the next smallest value, toward negative infinity.
        """
        if self._c == 0:
            if self._sided_interval:
                if self._negative:
                    return self.away(const_p=const_p, strict=strict)
                else:
                    return -self
            else:
                if strict:
                    sided = self._sided_interval
                else:
                    sided = False
                return Sink(self, e=self._n+1, p=1, c=1, negative=True, sided_interval=sided)
        elif self._negative:
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
        if center and self._sided_interval and self._inexact:
            extra_bits = 1 if self._full_interval else 2
            return Sink(self.narrow(n=self._n - extra_bits), inexact=False, sided_interval=False).away()
        else:
            return Sink(self, inexact=False)


    def explode(self, sided=None, full=None):
        """Explode a representative point to an enclosing interval.
        If provided, sided and full replace the corresponding properties of the original interval.
        It is invalid to explode a larger interval to a smaller one, i.e. full to half or
        unsided to sided.
        """
        if self._inexact:
            if sided and (not self._sided_interval):
                raise ValueError('explode: cannot shrink unsided interval {} to sided'.format(repr(self)))
            elif full and (not self._full_interval):
                raise ValueError('explode: cannot shrink full interval {} to half'.format(repr(self)))

        sided = self._sided_interval if sided is None else sided
        full = self._full_interval if full is None else full
        return Sink(self, inexact=True, sided_interval=sided, full_interval=full)


    def bounds(self):
        """Upper and lower bounds on the value of this number.
        Intervals are inclusive.
        """
        if self._inexact:
            if self._full_interval:
                base = self
            else:
                base = self.narrow(n=self._n - 1)

            if self._sided_interval:
                if self._negative:
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

        if self._n == n:
            return Sink(self)
        else:
            if self._n < n:
                # get rid of bits
                offset = n - self._n
                c = self._c >> offset
            else:
                # add bits
                offset = self._n - n
                c = self._c << offset
            # figure out p and e again
            p = c.bit_length()
            e = n + p
            return Sink(self, e=e, n=n, p=p, c=c)


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
            n = self._n
        offset = n - self._n

        if offset <= 0:
            if offset == 0 or self._inexact:
                return (
                    Sink(self, inexact=False),
                    Sink(0, n=self._n, negative=self._negative, inexact=self._inexact,
                         sided_interval=self._sided_interval, full_interval=self._full_interval),
                )
            else:
                extended_c = self._c << -offset
                extended_p = extended_c.bit_length()
                extended_e = n if extended_c == 0 else self._e
                return (
                    Sink(self, e=extended_e, n=n, p=extended_p, c=extended_c),
                    Sink(0, n=n, negative=self._negative, inexact=self._inexact,
                         sided_interval=self._sided_interval, full_interval=self._full_interval),
                )
        else:
            lost_bits = self._c & bitmask(offset)
            left_bits = self._c >> offset
            low_bits = lost_bits & bitmask(offset - 1)
            half_bit = lost_bits >> (offset - 1)

            e = max(self._e, n)
            inexact = self._inexact or lost_bits != 0
            if left_bits == 0:
                sided = True
            else:
                sided = self._sided_interval

            rounded = Sink(self, e=e, n=n, p=e-n, c=left_bits, inexact=False, sided_interval=sided)
            # in all cases we copy the sign onto epsilon... is that right?
            epsilon = Sink(0, n=n, negative=self._negative, inexact=inexact, sided_interval=sided, full_interval=self._full_interval)

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
            n = self._n
        else:
            n = min_n

        if max_p is not None:
            n = max(n, self._e - max_p)

        rounded, epsilon = self.split(n)

        if max_p is not None and rounded._p > max_p:
            # If we rounded up and carried, we might have increased p by one.
            # Split again to compensate; this should produce an epsilon of zero.
            rounded, epsilon_correction = rounded.split(n + 1)
            if not epsilon_correction.is_exactly_zero():
                epsilon = epsilon_correction
                raise ValueError('widen: unreachable')

        rounded._inexact = epsilon._inexact
        return rounded


    def narrow(self, n=None, p=None):
        """Force this number into a representation with either n or p.
        By default, preserve n and p, returning this number unchanged.
        Note that this may produce a smaller envelope that does not contain
        the input value.
        """
        if n is p is None:
            return Sink(self)
        elif n is None:
            if self._c == 0:
                # specifying precision is meaningless for zero
                n = self._n
            else:
                n = self._e - p
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

        if rounded._n > n:
            # split was unable to provide a small enough n, so we have to force one
            extended_c = rounded._c << (rounded._n - n)
            extended_p = extended_c.bit_length()
            extended_e = n if extended_c == 0 else rounded._e
            rounded = Sink(rounded, e=extended_e, n=n, p=extended_p, c=extended_c)
        elif p is not None and rounded._p > p:
            # as for widening, round again to compensate
            rounded, epsilon_correction = rounded.split(n+1)
            if not epsilon_correction.is_exactly_zero():
                epsilon = epsilon_correction
                raise ValueError('narrow: unreachable')

        rounded._inexact = epsilon._inexact
        return rounded


    def ieee_754(self, w, p):
        emax = (1 << (w - 1)) - 1
        emin = 1 - emax
        max_p = p
        min_n = emin - p

        if self._c == 0:
            return self.narrow(n=min_n)
        elif self._n <= min_n or self._p <= max_p:
            return self.widen(min_n=min_n, max_p=max_p)
        else:
            extra_bits = p - self._p
            return self.narrow(n=max(min_n, self._n - extra_bits))


    def to_mpfr(self):
        if self._negative:
            return conversion.mpfr_from_mantissa_exp(-self._c, self._n + 1)
        else:
            return conversion.mpfr_from_mantissa_exp(self._c, self._n + 1)


    def to_float(self, ftype=float):
        if ftype == float:
            w = 11
            p = 53
        elif ftype == np.float16:
            w = 5
            p = 11
        elif ftype == np.float32:
            w = 8
            p = 24
        elif ftype == np.float64:
            w = 11
            p = 53
        else:
            raise TypeError('expected float or np.float{{16,32,64}}, got {}'.format(repr(type(f))))

        rounded = self.ieee_754(w, p)

        if rounded._negative:
            return conversion.float_from_mantissa_exp(-rounded._c, rounded._n + 1, ftype=ftype)
        else:
            return conversion.float_from_mantissa_exp(rounded._c, rounded._n + 1, ftype=ftype)


    # core arith and comparison


    def __neg__(self):
        return Sink(self, negative=not self._negative)


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
        n = min(upper._n, lower._n, xupper._n, xlower._n)
        lower = lower.narrow(n=n)
        upper = upper.narrow(n=n)
        xlower = xlower.narrow(n=n)
        xupper = xupper.narrow(n=n)

        # convert to ordinals
        lower_ord = -lower._c if lower._negative else lower._c
        upper_ord = -upper._c if upper._negative else upper._c
        xlower_ord = -xlower._c if xlower._negative else xlower._c
        xupper_ord = -xupper._c if xupper._negative else xupper._c

        # integer comparison
        if not (lower_ord <= upper_ord and xlower_ord <= xupper_ord):
            # TODO: assertion
            self.details()
            x.details()
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

            n = min(rep._n, xrep._n)
            rep = rep.narrow(n=n)
            xrep = xrep.narrow(n=n)
            rep_ord = -rep._c if rep._negative else rep._c
            xrep_ord = -xrep._c if xrep._negative else xrep._c

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
        if self._inexact or x._inexact:
            raise ValueError('linear: can only add exact sinks, got {} + {}'.format(repr(self), repr(x)))
        n = min(self._n, x._n)
        c_norm = self._c << (self._n - n)
        xc_norm = x._c << (x._n - n)
        sign = -1 if self._negative else 1
        xsign = -1 if x._negative else 1
        signed_c = (sign * c_norm) + (xsign * xc_norm)
        if signed_c >= 0:
            c = signed_c
            negative = False
        else:
            c = -signed_c
            negative = True

        p = c.bit_length()
        e = n + p
        #TODO: inf and nan
        #TODO: sign of negative 0
        #TODO: envelope properties
        return Sink(self, e=e, n=n, p=p, c=c, negative=negative, sided_interval=False)


    def __sub__(self, x):
        """Alias of self + (-x)"""
        return self + (-x)


    def __mul__(self, x):
        """Multiply this sink by another sink x, exactly. Fails if either is inexact."""
        if self._inexact or x._inexact:
            raise ValueError('poly: can only multiply exact sinks, got {} * {}'.format(repr(self), repr(x)))
        n = self._n + x._n + 1 # equivalent to (self._n + 1) + (x._n + 1) - 1
        c = self._c * x._c
        p = c.bit_length()
        e = n + p
        #TODO: None in xor
        negative = bool(self._negative) != bool(x._negative)
        #TODO: inf and nan
        #TODO: envelope properties
        return Sink(self, e=e, n=n, p=p, c=c, negative=negative, sided_interval=False)
