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


def mask(n: int) -> int:
    """Produces a bitmask of n 1s."""
    return ~(-1 << n)


def mask_from(start: int, end: int) -> int:
    """Produces a bitmask of 1s from start to end."""
    return (~(-1 << (end - start))) << start


def msb(x: int) -> int:
    return x.bit_length()


def floorlog2(x: int) -> int:
    return max(x.bit_length() - 1, 0)


def _de_bruijn(k: int, n: int) -> typing.List[int]:
    """Generate a De Bruijn sequence.

    This is a piece of mathematical heavy machinery needed for O(1) clz on
    integers of bounded size. The algorithm is adapted from chapter 7 of
    Frank Ruskey's "Combinatorial Generation".

    Args:
        k: The number of symbols in the alphabet. Should be > 1.
        n: The length of subsequences. Should be > 1.

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


def _init_clz_table(n: int) -> typing.Tuple[int, typing.List[int]]:
    """Setup for fast clz of ints with up to 2**n bits.

    This code initializes the De Bruijn sequence (as an integer) and
    the lookup table needed to do constant time (1 multiply and some
    bit twiddling) clz, as described in Philib Busch's
    "Computing Trailing Zeros HOWTO".

    See: http://7ooo.mooo.com/text/ComputingTrailingZerosHOWTO.pdf
    """
    nbits = 1 << n
    nmask = mask(nbits)

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
# Note that _CTZ_DB_LENGTH cannot be changed at runtime unless the
# De Bruijn sequence and table being used are also updated.
_CTZ_DB_LENGTH = 6

# These constants are used internally by the ctz algorithm. They are
# generated based on _CTZ_DB_LENGTH.
_CTZ_DB_THRESH = 1 << (1 << _CTZ_DB_LENGTH)
_CTZ_DB_SHIFT = (1 << _CTZ_DB_LENGTH) - _CTZ_DB_LENGTH
_CTZ_DB_MASK = mask(_CTZ_DB_LENGTH)
_CTZ_DB, _CTZ_DB_TABLE = _init_clz_table(_CTZ_DB_LENGTH)

# For this to work at all, the De Bruijn sequence we're using must start
# with _CTZ_DB_LENGTH zeros. All sequences returned by our generator should
# have this property, but just in case let's check.
assert(_CTZ_DB >> _CTZ_DB_SHIFT == 0)

def ctz(x: int) -> typing.Tuple[int, int]:
    """Count trailing zeros.

    Counts the zeros on the least-significant end of the binary representation
    of x.

    Args:
        x: An int.

    Returns:
        A tuple (zeros, x >> zeros) where zeros is the number of trailing
        zeros in x.

    >>> ctz(0)
    (0, 0)
    >>> ctz(1)
    (0, 1)
    >>> ctz(2)
    (1, 1)
    >>> ctz(40)
    (5, 3)
    >>> ctz(-40)
    (-5, 3)
    >>> ctz(1 << 100)
    (100, 1)

    Should be fast for all conceivable integers.
    """
    if x < _CTZ_DB_THRESH:
        zeros = _CTZ_DB_TABLE[
            (((x & -x) * _CTZ_DB) >> _CTZ_DB_SHIFT) & _CTZ_DB_MASK
        ]
        return zeros, x >> zeros
    else:
        return -1, x


def naive_ctz(x: int) -> typing.Tuple[int, int]:
    """Count trailing zeros, naively.

    The behavior should be identical to ctz(x). This is a slow reference
    implementation for checking correctness.
    """
    zeros = 0
    while x & 1 == 0 and x > 0:
        x = x >> 1
        zeros += 1
    return zeros, x
