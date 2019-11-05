"""Universal representation for digital numbers (in base 2)"""

import typing
from enum import IntEnum, unique

from . import utils
from .ops import RM


@unique
class RoundingMode(IntEnum):
    TOWARD_ZERO = 0
    AWAY_ZERO = 1
    TO_EVEN = 2

@unique
class RoundingDirection(IntEnum):
    ROUND_DOWN = -1 # unused
    TRUNCATE = 0
    ROUND_AWAY = 1


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
    _rounded: bool = False

    # MPRF-like result code.
    # 0 if value is exact, -1 if it was rounded away, 1 if was rounded toward zero.
    # TODO: this field has been replaced with the rounding envelope info,
    # and is no longer supported.
    _rc: int = 0

    # rounding envelope
    _interval_size: int = 0
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
    def rounded(self):
        """Was this value rounded?
        Note that this is different from being inexact:
        A number is considered rounded if it was produced by some operation
        that could have produced more precision, and some of that extra precision
        was then rounded away.
        Thus, a rounded number is always inexact, but an inexact number
        may not actually be rounded (for example, if it was produced by
        subtraction of two inexact numbers of similar magnitude).
        """
        return self._rounded

    @property
    def interval_size(self):
        """Size of the rounding envelope.
        The envelope use to round to this number was 2 ** _interval_size
        units in the last place. For example:
        Fixed-point rounding, not to nearest, will have a size of 0.
        Fixed-point rounding, to nearest, will have a size of -1.
        Floating-point rounding, to nearest, which also round the significand
        up to a power of two, (thereby increasing _exp to keep precision the same)
        will have a size of -2.
        Other sizes are possible; in general, positive sizes don't make sense,
        and will cause further rounding to fail (the envelope is bigger than a whole ulp,
        so we have no idea which number this was supposed to be)
        while smaller negative sizes indicate that there was a known run of zeros or ones
        at the end of the significand.
        """
        return self._interval_size

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

    def is_zero(self):
        """Is this value a "classic" floating-point zero, with a zero significand?"""
        return self._c == 0

    def is_exactly_zero(self):
        """Is this value exactly zero, with a zero significand and the inexact flag unset?"""
        return self._c == 0 and (not self._inexact)

    def is_nonzero(self):
        """Is this value definitely not zero, with a nonzero significand?"""
        return self.c != 0

    def is_possibly_nonzero(self):
        """Is this value possibly nonzero, with a nonzero significand or the inexact flag set?"""
        return (self._c != 0) or self._inexact

    def is_integer(self):
        """Is this value an integer (though not necessarily an exact one)?"""
        return (self._exp >= 0) or (utils.maskbits(self._c, -self._exp) == 0)

    def is_exact_integer(self):
        """Is this value an exact integer?"""
        return (not self._inexact) and ((self._exp >= 0) or (utils.maskbits(self._c, -self._exp) == 0))

    def is_finite_real(self):
        """Is this value a finite real number, i.e. not an infinity or NaN?"""
        return not (self._isinf or self._isnan)

    def is_nar(self):
        """Is this value "not a real" number, i.e. an infinity or NaN?"""
        return self._isinf or self._isnan

    def is_identical_to(self, other):
        """Is this value encoded identically to some other value?
        This is a structural property, and may be stricter than real valued equality.
        """
        return (
            self._c == other._c
            and self._exp == other._exp
            and self._negative == other._negative
            and self._isinf == other._isinf
            and self._isnan == other._isnan
            and self._inexact == other._inexact
            and self._rounded == other._rounded
            and self._rc == other._rc # TODO remove me
            and self._interval_size == other._interval_size
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
                 rounded=None,
                 rc=None, # TODO remove me
                 interval_size=None,
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

        # _rounded
        if rounded is not None:
            self._rounded = rounded
        elif x is not None:
            self._rounded = x._rounded
        else:
            self._rounded = type(self)._rounded

        # _rc TODO remove me
        if rc is not None:
            self._rc = rc
        elif x is not None:
            self._rc = x._rc
        else:
            self._rc = type(self)._rc

        # interval stuff
        if interval_size is not None:
            self._interval_size = interval_size
        elif x is not None:
            self._interval_size = x._interval_size
        else:
            self._interval_size = type(self)._interval_size

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
        return '{}(negative={}, c={}, exp={}, inexact={}, rounded={}, rc={}, isinf={}, isnan={}, interval_size={}, interval_down={}, interval_closed={})'.format(
            type(self).__name__, repr(self._negative), repr(self._c), repr(self._exp),
            repr(self._inexact), repr(self._rounded), repr(self._rc), repr(self._isinf), repr(self._isnan),
            repr(self._interval_size), repr(self._interval_down), repr(self._interval_closed),
        )

    def __str__(self):
        return '{:s} {:d} * {:d}**{:d}{:s}{:s}'.format(
            '-' if self.negative else '+',
            self.c,
            self.base,
            self.exp,
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
                    raise utils.PrecisionError('cannot compare {} and {} with certainty'
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
                raise utils.PrecisionError('cannot compare {} and {} with certainty'
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
        TODO: This rounding method has been replaced by the new multi-method rounding
        code, and is no longer supported.
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
                raise utils.PrecisionError('cannot precisely round {} to p={}, n={}'.format(str(self), str(max_p), str(min_n)))
            else:
                # If the number is exact, then we can always extend with zeros. This is independent
                # of the rounding mode.
                # Shrinking the envelope invalidates the rc, so set it to 0. We know longer
                # know anything useful.
                return type(self)(self, c=self.c << -offset, exp=self.exp + offset, rc=0)

        # Break up the significand
        lost_bits = utils.maskbits(self.c, offset)
        left_bits = self.c >> offset

        if offset > 0:
            offset_m1 = offset - 1
            low_bits = utils.maskbits(lost_bits, offset_m1)
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

    # Most of the work done in the first phase is separated out into another
    # function that recovers what we know about the original significand.
    # The first and last phases are independent of the rounding mode.

    def round_recover(self):
        """Recover the information that was used to round to this number.
        This will give back a modified significand and exponent,
        as well as the low bit. The half bit (if it was known)
        is part of the new significand.
        Note that the precision of the new significand may change;
        in some cases, it could be less than the precision of the input significand
        (for example, fixed point rounding up to a power of two).

        The result is provided as a canonical significand and exponent c, exp
        such that the exact, pre_rounded number is exactly c * 2**exp if low_bit is 0,
        or somewhere (precise value unknown) between c * 2**exp and (c + 1) * 2**exp
        if low_bit is 1;
        i.e. c is the longest known prefix of the significand, rounded towards zero,
        and low_bit is the "sticky bit" which tells if there might be another 1
        somewhere in the lower bits of the significand.
        """
        if self.is_nar():
            raise utils.RoundingError('cannot recover rounding information from infinite or non-real values')

        if self.rounded:
            # Interval points towards zero, so we must have rounded UP.
            # The true significand is smaller.
            if self.interval_down:
                if self.is_zero():
                    raise utils.RoundingError('invalid interval for zero: cannot have rounded away from zero')
                # this subtraction may change the effective p and e, but not exp
                c = self.c - 1
                # now fix up based on interval size
                if self.interval_size < 0:
                    pad_len = -self.interval_size
                    pad = utils.bitmask(pad_len)
                    c = (c << pad_len) | pad
                    exp = self.exp - pad_len
                elif self.interval_size == 0:
                    exp = self.exp
                else: # self.interval_size > 0
                    raise utils.RoundingError('unsupported: interval size {} is larger than one ulp'
                                        .format(repr(self.interval_size)))

                # For closed intervals, use the modified significand exactly;
                # otherwise, we rounded up, so there must have been some low bits.
                if self.interval_closed:
                    low_bit = 0
                else:
                    low_bit = 1

            # Otherwise, interval points away from zero, so we rounded DOWN.
            # We may need to pad with zeros to extend the precision
            # based on how tight the interval is,
            # but we only change the value if the interval is known to be closed.
            else: # not self.interval_down
                c = self.c
                if self.interval_size < 0:
                    pad_len = -self.interval_size
                    # pad is zeros
                    c = c << pad_len
                    exp = self.exp - pad_len
                elif self.interval_size == 0:
                    exp = self.exp
                else: # self.interval_size > 0
                    raise utils.RoundingError('unsupported: interval size {} is larger than one ulp'
                                        .format(repr(self.interval_size)))

                # For closed intervals, we need to move one ulp away
                # (of the newly modified size); otherwise, there must have been some lower bits.
                if self.interval_closed:
                    # this addition may change the effective p and e (further), but not exp
                    c += 1
                    low_bit = 0
                else:
                    low_bit = 1

        else: # not self.rounded
            # This result wasn't rounded, so return exactly the information we have.
            # Even if the value is inexact, this is the best we can do.
            c = self.c
            exp = self.exp
            low_bit = 0

        return c, exp, low_bit

    def round_setup(self, max_p=None, min_n=None, ext_fn=None):
        """Split the significand in preparation for rounding.
        Will fail for any value that cannot round_recover(),
        specifically infinities and NaN.

        The result is the precision p (or none, if using fixed-point style rounding),
        as well as the exponent, and the split significand: c, the half bit, and the low bit.
        """
        c, exp, low_bit = self.round_recover()

        # compute p, n, and offset

        if max_p is None:
            p = None
            if min_n is None:
                # How are we supposed to round???
                raise ValueError('must specify max_p or min_n')
            else: # min_n is not None
                # Fixed-point rounding: limited by n, precision can change.
                n = min_n
        else: # max_p is not None:
            p = max_p
            e = (exp - 1) + c.bit_length()
            if min_n is None:
                # Floating-point rounding: limited by some fixed precision.
                n = e - max_p
            else: # min_n is not None
                # Floating-point rounding, with subnormals:
                # limited by some fixed precision, or a smallest representable bit.
                n = max(min_n, e - max_p)

        offset = n - (exp - 1)

        # split significand

        # Round off offset bits.
        if offset > 0:
            left_bits = c >> offset
            half_bit = (c >> (offset - 1)) & 1
            lost_bits = utils.maskbits(c, offset - 1)

            c = left_bits
            exp += offset

            if low_bit != 0 or lost_bits != 0:
                low_bit = 1
            else:
                low_bit = 0

        # Keep all of the bits; this means there can't be any half_bit.
        elif offset == 0:
            if self.inexact:
                half_bit = None
            else:
                half_bit = 0

        # Add on -offset bits to the right;
        # we're trying to make the number more precise.
        else: # offset < 0
            if ext_fn is None:
                if low_bit != 0:
                    raise utils.PrecisionError('cannot precisely split {} for rounding with p={}, n={}'
                                         .format(repr(self), repr(max_p), repr(min_n)))
                else:
                    # extend with zeros, which is entirely fine for exact values
                    c <<= -offset
                    exp += offset
                    # All information about the envelope is destroyed,
                    # so we always end up creating an exact value.
                    half_bit = 0
                    low_bit = 0
            else:
                raise ValueError('unsupported: ext_fn={}'.format(repr(ext_fn)))

        # TODO: sanity check
        assert exp == n+1
        assert p is None or min_n is not None or c == 0 or (c.bit_length() == p)

        return p, exp, c, half_bit, low_bit

    def round_direction(self, p, exp, c, half_bit, low_bit,
                        nearest=True, mode=RoundingMode.TO_EVEN):
        """Determine which direction to round, based on two criteria:
            - nearest, which determines if we should round to nearest when possible.
            - mode, which determines which way to break ties if rounding to nearest,
              or which direction to round in general otherwise.
        Returns the direction, the size of the interval (which is purely based on nearest:
        -1 if nearest is True, otherwise 0), and whether the interval is closed
        (which is only true for exact halfway values when rounding to nearest).
        """
        interval_closed = False
        if nearest:
            interval_size = -1
            if half_bit is None:
                raise utils.PrecisionError('insufficient precision to round {} ({} * 2**{}) to nearest with p={}'
                                     .format(repr(self), repr(c), repr(exp), repr(p)))
            # below half: truncate
            if half_bit == 0:
                direction = RoundingDirection.TRUNCATE
            else: # half_bit == 1
                # above half: round away
                if low_bit != 0:
                    direction = RoundingDirection.ROUND_AWAY
                # exactly halfway
                else: # low_bit == 0 and half_bit == 1
                    interval_closed = True
                    if mode is RoundingMode.TOWARD_ZERO:
                        direction = RoundingDirection.TRUNCATE
                    elif mode is RoundingMode.AWAY_ZERO:
                        direction = RoundingDirection.ROUND_AWAY
                    elif mode is RoundingMode.TO_EVEN:
                        if utils.is_even_for_rounding(c, exp):
                            direction = RoundingDirection.TRUNCATE
                        else:
                            direction = RoundingDirection.ROUND_AWAY
                    else:
                        raise ValueError('unknown rounding mode: {}'.format(repr(mode)))

        else: # not nearest
            interval_size = 0
            if mode is RoundingMode.TOWARD_ZERO:
                direction = RoundingDirection.TRUNCATE
            elif mode is RoundingMode.AWAY_ZERO:
                if (low_bit != 0) or ((half_bit is not None) and (half_bit != 0)):
                    direction = RoundingDirection.ROUND_AWAY
                else: # don't round away if we already have the value represented exactly!
                    direction = RoundingDirection.TRUNCATE
            elif mode is RoundingMode.TO_EVEN:
                if (low_bit != 0) or ((half_bit is not None) and (half_bit != 0)):
                    if utils.is_even_for_rounding(c, exp):
                        direction = RoundingDirection.TRUNCATE
                    else:
                        direction = RoundingDirection.ROUND_AWAY
                else:
                    direction = RoundingDirection.TRUNCATE
            else:
                raise ValueError('unknown rounding mode: {}'.format(repr(mode)))

        return direction, interval_size, interval_closed

    def round_apply(self, p, exp, c, half_bit, low_bit,
                    direction, interval_size, interval_closed):
        """Apply a rounding direction, to produce a rounded result."""
        if direction is RoundingDirection.ROUND_AWAY:
            # A considerable amount of though should be given to this behavior
            # in the edge cases that c is 1 or zero, i.e. we are rounding to less
            # than two bits of precision, which is not well defined for IEEE 754.
            #
            # If c is zero, we will round away to one. For fixed-point, this is the
            # right behavior. If we requested zero precision explicitly, we will
            # then notice that the precision has increased, chop it off back to zero,
            # but widen the exponent, which is also what we want.
            #
            # TODO: for p=0, do we still want to subtract from interval_size???
            #
            # If c is one, we will round away to two. For fixed-point, this is
            # right. If we requested one bit of precision explicitly, we will again
            # notice that the precision has increased, chop it off back to 1,
            # and increase the exponent instead, which is fine.
            c += 1
            rounded = True
            if p is not None and c.bit_length() > p:
                c >>= 1
                exp += 1
                interval_size -= 1
            interval_down = True

        elif direction is RoundingDirection.TRUNCATE:
            # Truncation doesn't have any special edge cases:
            # zero and one just stay as zero or one,
            # and the only interesting thing is that we have to check
            # the low_bit and half_bit to see if anything got rounded off
            # to make the operation inexact.
            rounded = (low_bit != 0) or (half_bit is not None and half_bit != 0)
            interval_down = False

        elif direction is RoundingDirection.ROUND_DOWN:
            raise RoundingError('rounding to the previous value is not supported')

        else:
            raise ValueError('unknown rounding direction: {}'.format(repr(direction)))

        # inexactness is only modified inderectly, if we seem to have rounded this number
        return type(self)(self, c=c, exp=exp, inexact=(self.inexact or rounded), rounded=rounded,
                          interval_size=interval_size, interval_down=interval_down, interval_closed=interval_closed)

    # negative, RM -> nearest, mode
    _rounding_modes = {
        (True, RM.RNE): (True, RoundingMode.TO_EVEN),
        (False, RM.RNE): (True, RoundingMode.TO_EVEN),
        (True, RM.RNA): (True, RoundingMode.AWAY_ZERO),
        (False, RM.RNA): (True, RoundingMode.AWAY_ZERO),
        (True, RM.RTP): (False, RoundingMode.TOWARD_ZERO),
        (False, RM.RTP): (False, RoundingMode.AWAY_ZERO),
        (True, RM.RTN): (False, RoundingMode.AWAY_ZERO),
        (False, RM.RTN): (False, RoundingMode.TOWARD_ZERO),
        (True, RM.RTZ): (False, RoundingMode.TOWARD_ZERO),
        (False, RM.RTZ): (False, RoundingMode.TOWARD_ZERO),
        (True, RM.RAZ): (False, RoundingMode.AWAY_ZERO),
        (False, RM.RAZ): (False, RoundingMode.AWAY_ZERO),
    }

    def round_new(self, max_p=None, min_n=None, rm=RM.RNE, strict=True):
        """Round the mantissa to at most max_p precision, or a least absolute digit
        in position min_n, whichever is less precise. Rounding is implemented generally
        for all real values; the requested precision may be one or even zero bits,
        but there is no limit on the resulting exponent, and infinite and NaN
        cannot be rounded in this way.

        If only min_n is given, then rounding is performed as for fixed-point,
        and the resulting significand may have more than max_p bits.
        If max_p is given, then rounding is performed as for floating-point,
        and the exponent will be adjusted to ensure the result has at most max_p bits.
        If both max_p and min_n are specified, then min_n takes precedence,
        so the result may have significantly less than max_p precision.
        This behavior can be used to emulate IEEE 754 subnormals.
        At least one of max_p or min_n must be given; otherwise, we would not know
        if the rounding behavior should use fixed-point or floating-point rules.

        The rounding mode rm is one of the usual IEEE 754 rounding modes.
        If strict is True, then rounding will raise a PrecisionError
        if there isn't enough precision to know the bits in the result.
        If strict is False, then the value will be converted to an exact
        value before rounding occurs; this ensures that there will not be a
        precision error, but may also change the result in some cases.

        Rounding may also raise a RoundingError if something else goes wrong;
        this is usually due to a nonsensical input, and turning strict off
        won't prevent it.
        """

        # round_setup will raise exceptions for bad max_p/min_n,
        # as well as for unroundable values like NaN.
        p, exp, c, half_bit, low_bit = self.round_setup(max_p=max_p, min_n=min_n)

        # convert the rounding mode and sign of this number into the internal rounding mode
        try:
            nearest, mode = Digital._rounding_modes[(self.negative, rm)]
        except KeyError:
            raise ValueError('invalid rounding mode: {} (negative={})'
                             .format(repr(rm), repr(self.negative)))

        # determine rounding direction and interval parameters
        direction, interval_size, interval_closed = self.round_direction(p, exp, c, half_bit, low_bit,
                                                                         nearest=nearest, mode=mode)

        # all done
        return self.round_apply(p, exp, c, half_bit, low_bit,
                                direction, interval_size, interval_closed)

    def next_float(self):
        """The next number with this precision, away from zero."""
        if self.is_nar():
            raise ValueError('there is no next float after {}'.format(repr(self)))

        next_c = self.c + 1
        next_exp = self.exp
        if next_c.bit_length() > self.p:
            next_c >>= 1
            next_exp += 1

        return type(self)(self, c=next_c, exp=next_exp, rounded=False)

    def prev_float(self):
        """The previous number with this precision, toward zero."""
        if self.is_nar():
            raise ValueError('there is no previous float before {}'.format(repr(self)))

        if self.c == 0:
            return type(self)(self, exp=self.exp-1, rounded=False)

        prev_c = self.c - 1
        prev_exp = self.exp
        if prev_c.bit_length() < self.p:
            prev_c = (prev_c << 1) | 1
            prev_exp -= 1

        return type(self)(self, c=prev_c, exp=prev_exp, rounded=False)
