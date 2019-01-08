"""Universal representation for digital numbers (in base 2)"""

import typing

from .ops import RM
from .integral import bitmask

class PrecisionError(Exception):
    """Insufficient precision given to rounding operation."""


class Digital(object):

    # for numbers with a real value, the magnitude is exactly _c * (_base ** _exp)
    _c : int = 0
    _exp : int = 0
    # base is always 2.

    # the sign is stored separately
    _negative : bool = False

    # as is information about infiniteness or NaN
    _isinf : bool = False
    _isnan : bool = False

    # the internal state is not directly visible: expose it with properties

    @property
    def c(self):
        """Unsigned integer significand.
        The magnitude of the real value is exactly (c * base**exp).
        """
        return self._c

    @property
    def exp(self):
        """Signed integer exponent.
        The magnitude of the real value is exactly (c * base**exp).
        """
        return self._exp

    @property
    def base(self):
        """Unsigned integer base. Must be >= 2. Always 2.
        The magnitude of the real value is exactly (m * base**exp).
        """
        return 2

    @property
    def m(self):
        """Signed integer significand.
        The real value is exactly (m * base**exp).
        """
        if self._negative:
            return -self._c
        else:
            return self._c

    @property
    def e(self):
        """IEEE 754 style exponent.
        If the significand is interpreted as a binary fraction between 1 and 2,
        i.e. x = 0b1.100101001110... etc. then the real value is (x * 2**e).
        """
        return (self._exp - 1) + self._c.bit_length()

    @property
    def n(self):
        """The "sticky bit" or the binary place where digits are no longer significant.
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

    # exactness
    _inexact: bool = False

    # MPRF-like result code.
    # 0 if value is exact, -1 if it was rounded away, 1 if was rounded toward zero.
    _rc: int = 0

    # rounding envelope
    _interval_full: bool = False
    _interval_down: bool = False
    _interval_closed: bool = False

    @property
    def inexact(self):
        """Is this vaue inexact?"""
        return self._inexact

    @property
    def rc(self):
        """Result code. Similar to the result code used by MPFR, but the values are different.
        If the rc is 0, this value was computed exactly.
        If the rc is -1, this value was computed inexactly, and rounded toward zero.
        If the rc is 1, this value was computed inexactly, and rounded away from zero.
        I.e. if the exact magnitude should be (c * base**exp) + (epsilon < base**n),
        then the rc gives the sign of the epsilon, or 0 if the epsilon is 0.
        This is now being replaced with the 3 interval flags.
        """
        return self._rc

    @property
    def interval_full(self):
        """Size of the rounding envelope.
        If True, then rounding was not to nearest, i.e. the interval covers
        the full space between this number and the next one with the same precision.
        If False, then rounding was to nearest, so the half bit can be recovered
        with certainty.
        """
        return self._interval_full

    @property
    def interval_down(self):
        """Direction of the rounding envelope.
        If True, then the interval points towards zero, i.e. this value is an
        upper bound on the true magnitude.
        If False, then the interval points away from zero.
        """
        return self._interval_down

    @property
    def interval_closed(self):
        """A proxy for the half bit.
        If true, then the interval includes the opposite endpoint.
        This isn't used for proper interval logic, but rather as a way to track
        round-nearest behavior in cases where a result exactly between two
        representable numbers has to be rounded, without losing information.
        """
        return self._interval_closed


    def is_exactly_zero(self):
        return self._c == 0 and not self._inexact

    def is_zero(self):
        return self._c == 0

    def is_integer(self):
        return self._exp >= 0 or self._c & bitmask(-self._exp) == 0

    def is_identical_to(self, other):
        return (
            self._c == other._c
            and self._exp == other._exp
            and self._negative == other._negative
            and self._isinf == other._isinf
            and self._isnan == other._isnan
            and self._inexact == other._inexact
            and self._rc == other._rc
            and self._interval_full == other._interval_full
            and self._interval_down == other._interval_down
            and self._interval_closed == other._interval_closed
        )

    def __init__(self,
                 x=None,
                 c=None,
                 negative=None,
                 m=None,
                 exp=None,
                 e=None,
                 isinf=None,
                 isnan=None,
                 inexact=None,
                 rc=None,
                 interval_full=None,
                 interval_down=None,
                 interval_closed=None,
    ):
        """Create a new digital number. The first argument, "x", is a base number
        to clone and update, otherwise the default values will be used.
        The significand can be specified as either c or m (if m is specified, then
        negative cannot be provided as an argument).
        The exponent can be specified as either exp or e. If it is specified as e,
        then the significand will first be set based on other arguments, then exp
        will be computed accordingly.
        """
        # _c and _negative
        if c is not None:
            if m is not None:
                raise ValueError('cannot specify both c={} and m={}'.format(repr(c), repr(m)))
            self._c = c
            if negative is not None:
                self._negative = negative
            elif x is not None:
                self._negative = x._negative
            else:
                self._negative = type(self)._negative
        elif m is not None:
            if negative is not None:
                raise ValueError('cannot specify both m={} and negative={}'.format(repr(m), repr(negative)))
            self._c = abs(m)
            self._negative = m < 0
        elif x is not None:
            self._c = x._c
            if negative is not None:
                self._negative = negative
            else:
                self._negative = x._negative
        else:
            self._c = type(self)._c
            if negative is not None:
                self._negative = negative
            else:
                self._negative = type(self)._negative

        # _exp
        if exp is not None:
            if e is not None:
                raise ValueError('cannot specify both exp={} and e={}'.format(repr(exp), repr(e)))
            self._exp = exp
        elif e is not None:
            self._exp = e - self._c.bit_length() + 1
        elif x is not None:
            self._exp = x._exp
        else:
            self._exp = type(self)._exp

        # _isinf
        if isinf is not None:
            self._isinf = isinf
        elif x is not None:
            self._isinf = x._isinf
        else:
            self._isinf = type(self)._isinf

        # _isnan
        if isnan is not None:
            self._isnan = isnan
        elif x is not None:
            self._isnan = x._isnan
        else:
            self._isnan = type(self)._isnan

        # _inexact
        if inexact is not None:
            self._inexact = inexact
        elif x is not None:
            self._inexact = x._inexact
        else:
            self._inexact = type(self)._inexact

        # _rc
        if rc is not None:
            self._rc = rc
        elif x is not None:
            self._rc = x._rc
        else:
            self._rc = type(self)._rc

        # interval stuff
        if interval_full is not None:
            self._interval_full = interval_full
        elif x is not None:
            self._interval_full = x._interval_full
        else:
            self._interval_full = type(self)._interval_full

        if interval_down is not None:
            self._interval_down = interval_down
        elif x is not None:
            self._interval_down = x._interval_down
        else:
            self._interval_down = type(self)._interval_down

        if interval_closed is not None:
            self._interval_closed = interval_closed
        elif x is not None:
            self._interval_closed = x._interval_closed
        else:
            self._interval_closed = type(self)._interval_closed

    def __repr__(self):
        return '{}(negative={}, c={}, exp={}, inexact={}, rc={}, isinf={}, isnan={}, interval_full={}, interval_down={}, interval_closed={})'.format(
            type(self).__name__, repr(self._negative), repr(self._c), repr(self._exp),
            repr(self._inexact), repr(self._rc), repr(self._isinf), repr(self._isnan),
            repr(self._interval_full), repr(self._interval_down), repr(self._interval_closed),
        )

    def __str__(self):
        return '{:s} {:d} * {:d}**{:d}{:s}{:s}{:s}'.format(
            '-' if self.negative else '+',
            self.c,
            self.base,
            self.exp,
            ' ~' + str(self.rc) if self.inexact else (' ' + str(self.rc) if self.rc else ''),
            ' inf' if self.isinf else '',
            ' nan' if self.isnan else '',
        )

    def compareto(self, other, strict=False):
        """Compare to another digital number. The ordering returned is:
            -1 iff self < other
             0 iff self = other
             1 iff self > other
          None iff self and other are unordered
        If strict is True, raise an exception for inexact numbers that cannot
        be ordered with certainty.
        """
        # deal with special cases
        if self.isnan or other.isnan:
            return None

        if self.isinf:
            if other.isinf and self.negative == other.negative:
                if strict and (self.inexact or other.inexact):
                    raise PrecisionError('cannot compare {} and {} with certainty'
                                         .format(str(self), str(other)))
                return 0
            elif self.negative:
                return -1
            else:
                return 1
        elif other.isinf:
            if other.negative:
                return 1
            else:
                return -1

        # normalize to smallest n - safe, but potentially inefficient
        n = min(self.n, other.n)

        # compare using ordinals
        self_ord = self.c << (self.n - n)
        other_ord = other.c << (other.n - n)

        if self.negative:
            self_ord = -self_ord
        if other.negative:
            other_ord = -other_ord

        if self_ord < other_ord:
            return -1
        elif self_ord == other_ord:
            if strict and (self.inexact or other.inexact):
                raise PrecisionError('cannot compare {} and {} with certainty'
                                     .format(str(self), str(other)))
            return 0
        else:
            return 1

    def __lt__(self, other):
        order = self.compareto(other)
        return order is not None and order < 0

    def __le__(self, other):
        order = self.compareto(other)
        return order is not None and order <= 0

    def __eq__(self, other):
        order = self.compareto(other)
        return order is not None and order == 0

    def __ne__(self, other):
        order = self.compareto(other)
        return order is None or order != 0

    def __ge__(self, other):
        order = self.compareto(other)
        return order is not None and order >= 0

    def __gt__(self, other):
        order = self.compareto(other)
        return order is not None and order > 0


    def round_m(self, max_p, min_n=None, rm=RM.RNE, strict=True):
        """Round the mantissa to at most max_p precision, or a least absolute digit
        in position min_n, whichever is less precise. Exact numbers can always be rounded
        to any precision, but rounding will fail if it would attempt to increase the
        precision of an inexact number (and strict is True). Rounding respects the rc,
        and sets it accordingly for the rounded result, so multiple rounding will never
        result in a loss of information. Rounding defaults to IEEE 754-style nearest even,
        with other modes not yet supported.
        """

        # some values cannot be rounded; clone but return unchanged
        if self.is_zero() or self.isinf or self.isnan:
            return type(self)(self)

        # determine where to round to, in terms of n
        if min_n is None:
            n = self.e - max_p
        else:
            n = max(min_n, self.e - max_p)

        offset = n - self.n

        if offset < 0:
            if strict and self.inexact:
                # If this number is inexact, then we'd have to make up bits to
                # extend the precision.
                raise PrecisionError('cannot precisely round {} to p={}, n={}'.format(str(self), str(max_p), str(min_n)))
            else:
                # If the number is exact, then we can always extend with zeros. This is independent
                # of the rounding mode.
                # Shrinking the envelope invalidates the rc, so set it to 0. We know longer
                # know anything useful.
                return type(self)(self, c=self.c << -offset, exp=self.exp + offset, rc=0)

        # Break up the significand
        lost_bits = self.c & bitmask(offset)
        left_bits = self.c >> offset

        if offset > 0:
            offset_m1 = offset - 1
            low_bits = lost_bits & bitmask(offset_m1)
            half_bit = lost_bits >> offset_m1
        else:
            # Rounding to the same precision is equivalent to having zero in the
            # lower bits; the only interesting information will come from the result code.
            low_bits = 0
            half_bit = 0

        # Determine which direction to round, based on rounding mode.
        #  1 := round away from zero
        #  0 := truncate towards zero
        # -1 := round down towards zero (this is very unusual)
        # Note that most rounding down will use truncation. Actual -1 direction
        # "round down" can only happen with 0 lost_bits and a contrary rc, i.e. we rounded
        # away but according to the new rounding mode we shouldn't have.
        # Zero cannot be given -1 direction: we can only keep it by truncation, or round away.
        direction = None

        if rm == RM.RNE:
            if half_bit == 0:
                # always truncate
                direction = 0
            else: # half_bit == 1
                if low_bits != 0:
                    # always round away
                    direction = 1
                else: # low_bits == 0
                    # break tie
                    if self.rc > 0:
                        direction = 1
                    elif self.rc < 0:
                        direction = 0
                    else: # rc == 0
                        if strict and self.inexact:
                            raise ValueError('unable to determine which way to round at this precision')
                        else: # not self.inexact
                            # round to even
                            if left_bits & 1 == 0:
                                direction = 0
                            else: # left_bits & 1 != 0
                                direction = 1
        else:
            raise ValueError('unimplemented: {}'.format(repr(rm)))

        c = left_bits
        exp = self.exp + offset
        inexact = self.inexact or (lost_bits != 0)

        if direction > 0:
            # round away
            c += 1
            if c.bit_length() > max_p:
                # we carried: shift over to preserve the right amount of precision
                c >>= 1
                exp += 1
            rc = -1
        elif direction == 0:
            # truncate
            if lost_bits != 0:
                # if some bits were truncated off, the result code should indicate a round down
                rc = 1
            else:
                # otherwise, preserve the old result code; nothing has changed
                rc = self.rc
        else: # direction < 0
            # round down, towards zero
            if direction is None:
                raise ValueError('no rounding direction ???')
            raise ValueError('unimplemented: round to previous')

        return type(self)(self, c=c, exp=exp, inexact=inexact, rc=rc)


    # Rounding is hard. We can break it up into 3 phases:
    #  - determine the target p and n, and split up the input
    #  - determine which direction to round
    #  - actually apply the rounding

    # The first and last phases are independent of the "rounding mode"

    def _round_setup(max_p, min_n=None, ext_fn=None, strict=True):
        """Determine p and n, and split the significand accordingly.
        Will fail for any inf or nan, as these non-real values cannot be rounded.
        If strict is True, will fail if the split cannot be performed precisely.
        """

        if self.isinf or self.isnan:
            raise ValueError('cannot round something that does not have a real value: {}'.format(repr(self)))

        # compute offset

        if min_n is None:
            n = self.e - max_p
        else:
            n = max(min_n, self.e - max_p)

        offset = n - self.n

        # Convert from envelope full / down / closed flags
        # to half bit / low bit, before shifting the significand.
        # This will be processed based on the offset later.

        c = self.c

        if self.inexact:

            if self.interval_full:
                known_half_bit = False
                half_bit = 0

                if self.interval_down:
                    if c == 0:
                        raise ValueError('inexact zero cannot have envelope down: {}'.format(repr(self)))
                    else:
                        c -= 1

                    if self.interval_closed:
                        # Closed intervals only occur when we round a point exactly on the border;
                        # here, we have rounded an exact value away to the next representable value.
                        # Note that this situation will never be produced by typical
                        # IEEE 754 rounding modes.
                        low_bit = 0
                    else: # not self.interval_closed
                        low_bit = 1

                else: # not self.interval_down
                    if self.interval_closed:
                        # We rounded an exact value towards zero, somehow.
                        c += 1
                        low_bit = 0
                    else: # not self.interval_closed
                        low_bit = 1

            else: # not self.interval_full
                known_half_bit = True

                if self.interval_down:
                    if c == 0:
                        raise ValueError('inexact zero cannot have envelope down: {}'.format(repr(self)))
                    else:
                        c -= 1

                    if self.interval_closed:
                        half_bit = 1
                        low_bit = 0
                    else: # not self.interval_closed
                        half_bit = 1
                        low_bit = 1

                else: # not self.interval_down
                    if self.interval_closed:
                        half_bit = 1
                        low_bit = 0
                    else:
                        half_bit = 0
                        low_bit = 1

        else: # not self.inexact
            known_half_bit = True
            half_bit = 0
            low_bit = 0




        if offset < 0:
            if strict and self.inexact:
                raise PrecisionError('cannot precisely split {} at p={}, n={}'.format(repr(self), repr(max_p), repr(min_n)))
            else:
                if ext_fn is None:
                    # extend with zeros
                    left_bits = self.c << -offset
                    # no bits are lost
                    lost_bits = 0
                    # all envelope information is destroyed, so we will always end up creating an exact value
                    known_half_bit = True
                    half_bit = 0
                    low_bit = 0
                else:
                    raise ValueError('unsupported: ext_fn={}'.format(repr(ext_fn)))

        elif offset == 0:
            left_bits = self.c
            lost_bits = 0

            # since we didn't lose any bits, try to recover low bits from the envelope
            if self.inexact:

                if self.interval_full:
                    known_half_bit = False
                    half_bit = 0

                    if self.interval_down:
                        if left_bits == 0:
                            raise ValueError('inexact zero cannot have envelope down: {}'.format(repr(self)))
                        else:
                            left_bits -= 1

                        if self.interval_closed:
                            # Closed intervals only occur when we round a point exactly on the border;
                            # here, we have rounded an exact value away to the next representable value.
                            # Note that this situation will never be produced by typical
                            # IEEE 754 rounding modes.
                            low_bit = 0
                        else: # not self.interval_closed
                            low_bit = 1

                    else: # not self.interval_down
                        if self.interval_closed:
                            # We rounded an exact value towards zero, somehow.
                            left_bits += 1
                            low_bit = 0
                        else: # not self.interval_closed
                            low_bit = 1

                else: # not self.interval_full

                    if self.interval_down:
                        known_half_bit = True

                        if left_bits == 0:
                            raise ValueError('inexact zero cannot have envelope down: {}'.format(repr(self)))
                        else:
                            left_bits -= 1

                        if self.interval_closed:
                            half_bit = 1
                            low_bit = 0
                        else: # not self.interval_closed
                            not self.interval_closed:
                            half_bit = 1
                            low_bit = 1

                    else: # not self.interval_down
                        if self.interval_closed:
                            half_bit = 1
                            low_bit = 0
                        else:
                            half_bit = 0
                            low_bit = 1

            # unless there is no envelope, in which case, do nothing
            else: # not self.inexact
                known_half_bit = True
                half_bit = 0
                low_bit = 0

        else: # offset > 0
            left_bits = self.c >> offset
            lost_bits = self.c & bitmask(offset)

            if offset == 1:
                half_bit

            if self.inexact:
                rc = self.rc
            else:
                # again, zero the rc if exact
                rc = 0

        return max_p, n, offset, left_bits, lost_bits, half_bit, low_bits, rc


    _TOWARD_ZERO = 0
    _AWAY_ZERO = 1
    _TO_EVEN = 2

    _ROUND_DOWN = -1
    _TRUNCATE = 0
    _ROUND_AWAY = 1

    def _round_direction(offset, left_bits, lost_bits, rc,
                         nearest=True, mode=_TO_EVEN, strict=True):
        """Determine which direction to round, based on two criteria:"
            - nearest, which determines if we should round to nearest when possible,
            - mode, which determines which way to break ties if rounding to nearest,
              or which direction to round in general otherwise.
        The _TO_EVEN mode is only supported when rounding to nearest.
        If strict is True, will fail if the direction cannot be determined precisely,
        i.e. if an inexact number on a boundary is given without a return code.
        """

        if offset <= 0:
            half_bit == 0
            low_bits == 0
        else:
            offset_m1 = offset - 1
            half_bit = lost_bits >> offset_m1
            low_bits = lost_bits & bitmask(offset_m1)

        if nearest:
            if half_bit == 0:
                return Digital._TRUNCATE
            else:
                if low_bits != 0:
                    return Digital._ROUND_AWAY
                else:
                    # note that if the number is exact, it must have rc 0
                    if rc < 0:
                        return Digital._TRUNCATE
                    elif rc > 0:
                        return Digital._ROUND_AWAY
                    else: # rc == 0
                        if strict and self.inexact:
                            raise ValueError('unable to determine which way to round at this precision')
                        # break tie
                        if mode == Digital._TOWARD_ZERO:
                            return Digital._TRUNCATE
                        elif mode == Digital._AWAY_ZERO:
                            return Digital._ROUND_AWAY
                        elif mode == Digital._TO_EVEN:
                            if left_bits & 1 == 0:
                                return Digital._TRUNCATE
                            else:
                                return Digital._ROUND_AWAY
                        else:
                            raise ValueError('Unknown rounding mode {}'.format(mode))

        else:
            if mode == Digital._TOWARD_ZERO:
                if lost_bits == 0:
                    # again, if the number is exact, then this rc is 0
                    if rc < 0:
                        # special case: this number was rounded away from zero, but now we want to
                        # round it toward zero, so undo the previous rounding by actually subtracting
                        return Digital._ROUND_DOWN
                    elif rc > 0:
                        return Digital._TRUNCATE
                    else: # rc == 0
                        if strict and self.inexact:
                            raise ValueError('unable to determine which way to round at this precision')
                        # assume exact case: just keep this number
                        return Digital._TRUNCATE
                else:
                    return Digital._TRUNCATE

            elif mode == Digital._AWAY_ZERO:
                if lost_bits == 0:
                    # if the number is exact, then rc must be 0, blah blah blah
                    if rc < 0:
                        # we rounded away to get here; keep it
                        return Digital._TRUNCATE
                    elif rc > 0:
                        return Digital._ROUND_AWAY
                    else: # rc == 0
                        if strict and self.inexact:
                            raise ValueError('unable to determine which way to round at this precision')
                        # assume exact case: just keep this number
                        return Digital._TRUNCATE
                else:
                    return Digital._ROUND_AWAY

            elif mode == Digital._TO_EVEN:
                raise ValueError('unsupported: nearest={}, mode={}'.format(nearest, mode))

            else:
                raise ValueError('Unknown rounding mode {}'.format(mode))


    def _round_apply(max_p, offset, left_bits, lost_bits, rc,
                     direction):
        """Apply a rounding direction, to produce a rounded result.
        Will fail if an attempt is made to _ROUND_DOWN zero; this direction should
        never be attempted in practice, as a zero cannot be computed with a
        negative return code (i.e. we can't have an inexact zero such that the
        true result was actually smaller than it).
        """

        c = left_bits
        exp = self.exp + offset

        if direction == Digital._ROUND_AWAY:
            c += 1
            if c.bit_length() > max_p:
                # we carried: shift over to preserve p
                c >>= 1
                exp += 1
            new_rc = -1

        elif direction == Digital._TRUNCATE:
            if lost_bits != 0:
                new_rc = 1
            else:
                new_rc = rc


        elif direction == Digital._ROUND_DOWN:
            if c == 0:
                raise ValueError('cannot round zero down')

            c -= 1
            if c.bit_length() < max_p:
                # we borrowed: shift back to preserve p
                c <<= 1
                exp -= 1
            new_rc = 1

        else:
            raise ValueError('Unknown direction {}'.format(direction))

        inexact = self.inexact or (rc != 0)

        return type(self)(self, c=c, exp=exp, inexact=inexact, rc=new_rc)
