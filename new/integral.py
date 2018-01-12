"""Integer utilities and bitvectors.

This file is part of Titanic, available under the MIT license.

The BV class implements bitvectors of known size. Bitvectors
can be shifted and concatenated, and bits or sub-vectors can be
extracted. They can be compared for equality, but not order,
and interpreted as (unsigned only?) integers.

Also provides small utilities are provided for integers:
  msb(x): "most significant bit" of x, as x.bit_length()
  floorlog2(x): x.bit_length() - 1, 0 if x is 0
  ctz(x): "count trailing zeros"
"""

import typing


class BV(object):
    pass


def bitmask(n: int) -> int:
    """Produces a bitmask of n 1s."""
    return ~(-1 << n)


def bitmask_at(start: int, end: int) -> int:
    """Produces a bitmask of 1s from start to end."""
    return (~(-1 << (end - start))) << start


def msb(x: int) -> int:
    return x.bit_length()


def floorlog2(x: int) -> int:
    return max(x.bit_length() - 1, 0)


def naive_ctz2(x: int) -> typing.Tuple[int, int]:
    """Count trailing zeros, naively.

    Args:
        x: An int.

    Returns:
        A tuple (zeros, x >> zeros) where zeros is the number of trailing
        zeros in x. I.e. (x >> zeros) & 1 == 1 or x == zeros == 0.

    This is a slow reference implementation for checking correctness. Since
    it never needs to check the bit_length of x or do any multiplication,
    it may be acceptably fast for numbers that have a very small number of
    trailing zeros.

    As it inherently tracks the remaining significant bits of x, this
    implementation returns them as well as the number of zeros, hence its
    name "ctz2".

    >>> naive_ctz2(0)
    (0, 0)
    >>> naive_ctz2(1)
    (0, 1)
    >>> naive_ctz2(-1)
    (0, -1)
    >>> naive_ctz2(2)
    (1, 1)
    >>> naive_ctz2(-2)
    (1, -1)
    >>> naive_ctz2(40) # 0b101000 = 2**3 * 5
    (3, 5)
    >>> naive_ctz2(-40) # 0b1..1011000
    (3, -5)
    >>> naive_ctz2(37 << 100)
    (100, 37)
    """
    if x == 0:
        return 0, 0
    else:
        zeros: int = 0
        while x & 1 == 0:
            x = x >> 1
            zeros += 1
        return zeros, x


def naive_sqrt_ctz2(x: int) -> typing.Tuple[int, int]:
    """Count trailing zeros, in O(sqrt(zeros)) steps.

    Args:
        x: An int.

    Returns:
        A tuple (zeros, x >> zeros) where zeros is the number of trailing
        zeros in x. I.e. (x >> zeros) & 1 == 1 or x == zeros == 0.

    This implementation is much faster than the naive linear implementation
    for many numbers of reasonable size. Because it looks at windows of
    linearly increasing size, it only needs to perform the square root
    of the number of zeros total steps, and it avoids constructing a number
    with many more bits than x (the logarithmic implementation below can
    compare x against a mask with twice as many bits in the worst case).
    Because it tracks the remaining significant bits of x explicitly, this
    implementation can return them more efficiently than the caller re-doing
    the right shift, so it is provided as a "ctz2".

    We still say this implementation is "naive" because it does not try
    to use any specific optimization tricks besides the sqrt algorithm. For
    some numbers with a very large number of zeros, it may be asymptotically
    worse than the logarithmic algorithm.

    >>> naive_sqrt_ctz2(0)
    (0, 0)
    >>> naive_sqrt_ctz2(1)
    (0, 1)
    >>> naive_sqrt_ctz2(-1)
    (0, -1)
    >>> naive_sqrt_ctz2(2)
    (1, 1)
    >>> naive_sqrt_ctz2(-2)
    (1, -1)
    >>> naive_sqrt_ctz2(40) # 0b101000 = 2**3 * 5
    (3, 5)
    >>> naive_sqrt_ctz2(-40) # 0b1..1011000
    (3, -5)
    >>> naive_sqrt_ctz2(37 << 100)
    (100, 37)

    # Of course the behavior should match for all integers...
    >>> all(naive_ctz2(x) == naive_sqrt_ctz2(x) for x in range(1024))
    True
    """
    if x == 0:
        return 0, 0
    else:
        zmask = 1
        zscale = 1
        zeros: int = 0

        # divide out larger and larger powers, until x % 2**zscale != 0
        while x & zmask == 0:
            x >>= zscale
            zeros += zscale
            zscale += 1
            zmask = (zmask << 1) | 1

        # recover smaller powers
        while zscale > 0:
            zmask >>= 1
            zscale -= 1
            if x & zmask == 0:
                x = x >> zscale
                zeros += zscale

        # Using for with a range may allow Python to produce more efficient
        # bytecode for this loop, but the difference is only a few percent,
        # at least based on some quick testing with CPython 3.6.3.

        # for zscale in range(zscale-1, 0, -1):
        #     zmask >>= 1
        #     if x & zmask == 0:
        #         x = x >> zscale
        #         zeros += zscale

        return zeros, x


def naive_log_ctz(x: int) -> int:
    """Count trailing zeros, in a O(log(zeros)) steps.

    Args:
        x: An int.

    Returns:
        The number of trailing zeros in x, as an int.

    This implementation is much faster than the naive linear implementation,
    as it performs a logarithmic number of steps relative to the number of
    trailing zeros in x. Unlike the linear implementation, this one avoids
    looking at the high bits of x if it can, so it only returns the number
    of zeros, not the remaining significant bits of x.

    We still say this implementation is "naive" because it does not try
    to use any specific optimization tricks besides the logarithmic algorithm.

    >>> naive_log_ctz(0)
    0
    >>> naive_log_ctz(1)
    0
    >>> naive_log_ctz(-1)
    0
    >>> naive_log_ctz(2)
    1
    >>> naive_log_ctz(-2)
    1
    >>> naive_log_ctz(40) # 0b101000 = 2**3 * 5
    3
    >>> naive_log_ctz(-40) # 0b1..1011000
    3
    >>> naive_log_ctz(37 << 100)
    100

    # Of course the behavior should match for all integers...
    >>> all(naive_ctz2(x)[0] == naive_log_ctz(x) for x in range(1024))
    True
    """
    if x == 0:
        return 0
    else:
        zmask = 1
        zscale = 1
        low_bits = x & zmask

        while low_bits == 0:
            zmask = (zmask << zscale) | zmask
            zscale <<= 1
            low_bits = x & zmask

        zscale >>= 1
        zmask >>= zscale
        zeros : int = 0

        while zscale > 0:
            if low_bits & zmask == 0:
                low_bits >>= zscale
                zeros += zscale
            zscale >>= 1
            zmask >>= zscale

        return zeros


def _de_bruijn(k: int, n: int) -> typing.List[int]:
    """Generate a De Bruijn sequence.

    This is a piece of mathematical heavy machinery needed for O(1) ctz on
    integers of bounded size. The algorithm is adapted from chapter 7 of
    Frank Ruskey's "Combinatorial Generation".

    Args:
        k: The number of symbols in the alphabet. Must be > 1.
        n: The length of subsequences. Should be >= 0.

    Returns:
        The De Bruijn sequence, as a list of integer indices.

    >>> _de_bruijn(2, 3)
    [0, 0, 0, 1, 0, 1, 1, 1]
    >>> _de_bruijn(4, 2)
    [0, 0, 1, 0, 2, 0, 3, 1, 1, 2, 1, 3, 2, 2, 3, 3]
    >>> _de_bruijn(2, 5)
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1]
    """
    a: typing.List[int] = [0] * k * n
    sequence: typing.List[int] = []

    def gen(t: int, p: int):
        if t > n:
            if n % p == 0:
                sequence.extend(a[1:p + 1])
        else:
            a[t] = a[t - p]
            gen(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                gen(t + 1, t)

    gen(1, 1)
    return sequence


def _init_ctz_db_and_table(n: int) -> typing.Tuple[int, typing.List[int]]:
    """Setup for fast ctz of ints with up to 2**n bits.

    This code initializes the De Bruijn sequence (as an integer) and
    the lookup table needed to do constant time (1 multiply and some
    bit twiddling) ctz, as described in Philib Busch's
    "Computing Trailing Zeros HOWTO".

    See: http://7ooo.mooo.com/text/ComputingTrailingZerosHOWTO.pdf

    The following weird edge cases are correct, though they aren't exactly
    useful except to turn the fast ctz algorithm off:
    >>> _init_ctz_db_and_table(0)
    (0, [0])
    >>> _init_ctz_db_and_table(1)
    (1, [0, 1])

    This initialization is too small to be much use:
    >>> _init_ctz_db_and_table(4)
    (2479, [0, 1, 2, 5, 3, 9, 6, 11, 15, 4, 8, 10, 14, 7, 13, 12])
    """
    nbits = 1 << n
    nmask = bitmask(nbits)

    db : int = 0

    sequence = _de_bruijn(2, n)
    # we want the first elements to go in the high bits, so reverse the order
    for i, bit in enumerate(reversed(sequence)):
        db |= bit << i

    table: typing.List[int] = [-1] * nbits

    shift_offset = nbits - n
    shifted_db = db
    for i in range(nbits):
        table[shifted_db >> shift_offset] = i
        # mask to nbits bits
        shifted_db = (shifted_db << 1) & nmask

    return db, table


# For small integers, it may be possible to perform ctz in constant
# time. This parameter control how small those integers have to be.
# With a bit of very quick tuning, it seems like we can set this at about
# 10 before the overhead of large multiplies starts to bite us, even for
# very small inputs.
_CTZ_DB_LENGTH: int = 10

# These constants are used internally by the ctz algorithm. They are
# generated based on _CTZ_DB_LENGTH.
_CTZ_DB_THRESH_BITS: int = 1 << _CTZ_DB_LENGTH
_CTZ_DB_SHIFT: int = (1 << _CTZ_DB_LENGTH) - _CTZ_DB_LENGTH
_CTZ_DB_MASK: int = bitmask(_CTZ_DB_LENGTH)
_INIT_CTZ_DB, _INIT_CTZ_DB_TABLE = _init_ctz_db_and_table(_CTZ_DB_LENGTH)
_CTZ_DB: int = _INIT_CTZ_DB
_CTZ_DB_TABLE : typing.List[int] = _INIT_CTZ_DB_TABLE

# We might as well avoid recomputing these values each time we call ctz.
_CTZ_ZMASK : int = bitmask(_CTZ_DB_THRESH_BITS)
_CTZ_ZSCALE : int = _CTZ_DB_THRESH_BITS

# For this to work at all, the De Bruijn sequence we're using must start
# with _CTZ_DB_LENGTH zeros. All sequences returned by our generator should
# have this property, but just in case let's check.
assert(_CTZ_DB >> _CTZ_DB_SHIFT == 0)


# For very small integers, or numbers with a small number of trailing zeros
# (such as normal numbers), we can get the number of trailing zeros efficiently
# with a lookup table.
_CTZ_SMALL_THRESH_BITS: int = 10

# The mask and table are generated based on the threshold bits. We store a table of size
# 2 ** _CTZ_SMALL_THRESH_BITS, and can use it for integers with their first set bit
# within the threshold.
_CTZ_SMALL_THRESH_MASK: int = bitmask(_CTZ_SMALL_THRESH_BITS)
_CTZ_SMALL_TABLE: typing.List[int] = [z for z, _ in (naive_ctz2(x) for x in range(1 << _CTZ_SMALL_THRESH_BITS))]

# Note that the parameters cannot be changed at runtime, unless the change
# manages to update all of the values atomically relative to any calls to
# ctz.


def ctz(x: int) -> int:
    """Count trailing zeros.

    Counts the zeros on the least-significant end of the binary representation
    of x.

    Args:
        x: An int.

    Returns:
        The number of trailing zeros as an int.

    >>> ctz(0)
    0
    >>> ctz(1)
    0
    >>> ctz(-2)
    1
    >>> ctz(40) # 2**3 * 5
    3
    >>> ctz(37 << 100)
    100

    # Of course the behavior should match for all integers...
    >>> all(naive_ctz2(x)[0] == ctz(x) for x in range(1024))
    True

    Should be fast for all conceivable integers.
    """
    low_bits = x & _CTZ_SMALL_THRESH_MASK
    # Test for x == 0 under a short circuit or, so it's off the control
    # flow path for the (probably common) case of nonzero x with a small
    # number of trailing zeros.
    if low_bits != 0 or x == 0:
        return _CTZ_SMALL_TABLE[low_bits]
    else:
        low_bits = x & _CTZ_ZMASK
        # use the De Bruijn sequence immediately if possible
        if low_bits != 0:
            return _CTZ_DB_TABLE[
                (((low_bits & -low_bits) * _CTZ_DB) >> _CTZ_DB_SHIFT) & _CTZ_DB_MASK
            ]
        else:
            zmask = _CTZ_ZMASK
            zscale = _CTZ_ZSCALE
            zeros: int = 0

            # The number of steps we need to do for the logarithmic algorithm
            # is tricky. What we want is to shift off large zero regions, until
            # the remaining region containing the rightmost 1 is small enough
            # to use the De Bruijn sequence.
            #
            # x.bit_length().bit_length() would be the number of bits required to
            # represent the number of bits in x, or log(log(x)). We actually care about
            # a slightly different quantity, (x.bit_length() - 1).bit_length(), which
            # is the number of round it takes to process x with the logarithmic algorithm,
            # since, for example with 1024 bits, we don't actually need to mask against
            # the full 1024; we know at this point that x is not 0, so we can skip
            # that round and actually start with 512.
            #
            # We iterate from _CTZ_DB_LENGTH, which tells us how many rounds we can pass
            # off to the De Bruijn sequence, up to this maximum that we've calculated
            # based on the size of x. Finally, we add 1 to _CTZ_DB_LENGTH because we've
            # already checked the first round above.
            #
            # For example, say _CTZ_DB_LENGTH is 10. The computed maximum for x with 2048
            # bits is 11. range(11, 11) is empty, so we won't check any other rounds;
            # this is good, because we don't actually need to, we can just shift off the
            # first 1024 zeros and use the De Bruijn sequence on the remaining 1024 bits.
            #
            # The computed maximum for x with 2049 bits is 12, however, so range(11, 12)
            # will contain one element, and we'll check if there's a 1 in the low 2048 bits.

            upstep = _CTZ_DB_LENGTH
            for upstep in range(_CTZ_DB_LENGTH + 1, (x.bit_length() - 1).bit_length()):
                #print('up', upstep)

                # scale first, as we've already tested with the initial mask
                zmask = (zmask << zscale) | zmask
                zscale <<= 1
                low_bits = x & zmask
                if low_bits != 0:
                    # only scale down if we break
                    zscale >>= 1
                    zmask >>= zscale
                    break

            # At this point, we maintain the invariant that the mask reflects the largest
            # mask of all zeros. If we went all the way through the loop without masking a 1,
            # then we know that the next larger mask would have more bits than x, and since
            # x is nonzero, it would have to mask a 1. If we broke out early because we
            # masked a 1, then we scaled down the mask before we did so.
            #
            # The loop variable from above, upstep, holds the number of the round that
            # corresponds to this zero mask, i.e. mask.bit_length().bit_length().
            # We need to iterate every value back down to and including _CTZ_DB_LENGTH,
            # which we can do with range(upstep, _CTZ_DB_LENGT - 1, -1).

            # if we made it through the loop, it's implied that low_bits would be all of x
            if low_bits == 0:
                low_bits = x

            #print(x, low_bits, upstep, zscale, zmask, zmask.bit_length())

            for downstep in range(upstep, _CTZ_DB_LENGTH - 1, -1):
                #print('down', downstep)

                if low_bits & zmask == 0:
                    low_bits >>= zscale
                    zeros += zscale
                zscale >>=1
                zmask >>= zscale

            # There must be few enough bits remaining at this point that we can use the
            # De Bruijn sequence.

            return zeros + _CTZ_DB_TABLE[
                (((low_bits & -low_bits) * _CTZ_DB) >> _CTZ_DB_SHIFT) & _CTZ_DB_MASK
            ]


def ctz_alt(x: int) -> int:
    """Count trailing zeros.

    Counts the zeros on the least-significant end of the binary representation
    of x.

    Args:
        x: An int. Must be >= 0.

    Returns:
        The number of trailing zeros.

    >>> ctz_alt(0)
    0
    >>> ctz_alt(1)
    0
    >>> ctz_alt(2)
    1
    >>> ctz_alt(40) # 2**3 * 5
    3
    >>> ctz_alt(37 << 100)
    100

    An early attempt at an optimized implementation.
    """
    if x.bit_length() <= _CTZ_DB_THRESH_BITS:
        return _CTZ_DB_TABLE[
            (((x & -x) * _CTZ_DB) >> _CTZ_DB_SHIFT) & _CTZ_DB_MASK
        ]
    else:
        zeros = 0
        f2 = 1
        fm = 1
        # divide out larger and larger powers, until x % 2**f2 != 0
        while x & fm == 0 and x > 0:
            x >>= f2
            zeros += f2
            f2 += 1
            fm = (fm << 1) | 1
        # recover smaller powers
        for f2 in range(f2-1, 0, -1):
            fm >>= 1
            if x & fm == 0:
                x = x >> f2
                zeros += f2
        return zeros


def ctz_split(x: int) -> typing.Tuple[int, int]:
    """Count trailing zeros, and separate out the interesting bits.

    As ctz_alt(x), but also reports the "significant" part of x, shifted over
    by the number of zeros. This is the same as factoring out all powers
    of 2 to produce z and y such that x = 2**z * y.

    Args:
        x: An int. Must be >= 0.

    Returns:
        A tuple (zeros, x >> zeros) where zeros is the number of trailing
        zeros in x.

    >>> ctz_split(0)
    (0, 0)
    >>> ctz_split(1)
    (0, 1)
    >>> ctz_split(2)
    (1, 1)
    >>> ctz_split(40) # 2**3 * 5
    (3, 5)
    >>> ctz_split(37 << 100)
    (100, 37)
    """
    if x.bit_length() <= _CTZ_DB_THRESH_BITS:
        zeros = _CTZ_DB_TABLE[
            (((x & -x) * _CTZ_DB) >> _CTZ_DB_SHIFT) & _CTZ_DB_MASK
        ]
        return zeros, x >> zeros
    else:
        zeros = 0
        f2 = 1
        fm = 1
        # divide out larger and larger powers, until x % 2**f2 != 0
        while x & fm == 0 and x > 0:
            x >>= f2
            zeros += f2
            f2 += 1
            fm = (fm << 1) | 1
        # recover smaller powers
        for f2 in range(f2-1, 0, -1):
            fm >>= 1
            if x & fm == 0:
                x = x >> f2
                zeros += f2
        return zeros, x


if __name__ == "__main__":
    import doctest
    import timeit

    def test_range(e, mbits):
        for m in range(1 << mbits):
            for shift in range(e):
                x = m << shift
                naive_z, naive_y = naive_ctz2(x)
                sqrt_z, sqrt_y = naive_sqrt_ctz2(x)
                log_z = naive_log_ctz(x)
                z, y = ctz_split(x)
                alt_z = ctz_alt(x)
                just_z = ctz(x)

                if not (naive_z == sqrt_z == log_z == z == alt_z == just_z):
                    print(x, naive_z, sqrt_z, log_z, z, alt_z, just_z)
                if not (naive_y == sqrt_y == y):
                    print(x, naive_y, sqrt_y, y)

    def time_naive(e, mbits):
        for m in range(1 << mbits):
            for shift in range(e):
                x = m << shift
                z, y = naive_ctz2(x)

    def time_sqrt(e, mbits):
        for m in range(1 << mbits):
            for shift in range(e):
                x = m << shift
                z, y = naive_sqrt_ctz2(x)

    def time_log(e, mbits):
        for m in range(1 << mbits):
            for shift in range(e):
                x = m << shift
                z = naive_log_ctz(x)

    def time_alt(e, mbits):
        for m in range(1 << mbits):
            for shift in range(e):
                x = m << shift
                z = ctz_alt(x)

    def time_fast(e, mbits):
        for m in range(1 << mbits):
            for shift in range(e):
                x = m << shift
                z = ctz(x)

    def time_blank(e, mbits):
        for m in range(1 << mbits):
            for shift in range(e):
                x = m << shift

    def compare_e_m(e, mbits, reps, naive=False):
        print('for e={:d}, mbits={:d}, {:d} reps'.format(e, mbits, reps))
        if naive:
            print('  naive: {:.3f}'.format(timeit.timeit(lambda: time_naive(e, mbits), number=reps)))
        print('  sqrt : {:.3f}'.format(timeit.timeit(lambda: time_sqrt(e, mbits), number=reps)))
        print('  log  : {:.3f}'.format(timeit.timeit(lambda: time_log(e, mbits), number=reps)))
        print('  alt  : {:.3f}'.format(timeit.timeit(lambda: time_alt(e, mbits), number=reps)))
        print('  fast : {:.3f}'.format(timeit.timeit(lambda: time_fast(e, mbits), number=reps)))
        print('  blank: {:.3f}'.format(timeit.timeit(lambda: time_blank(e, mbits), number=reps)))

    doctest.testmod()
    test_range(8301, 4)
    test_range(8, 17)

    compare_e_m(8, 17, 10, True)
    compare_e_m(8301, 4, 10, False)
