"""General utilities, such as exception classes."""

import typing

# Titanic-specific exceptions

class TitanicError(Exception):
    """Base Titanic error."""

class RoundingError(TitanicError):
    """Rounding error, such as attempting to round NaN."""

class PrecisionError(RoundingError):
    """Insufficient precision to perform rounding."""


# Useful things

def bitmask(n: int) -> int:
    """Produces a bitmask of n 1s if n is positive, or n 0s if n is negative."""
    if n >= 0:
        return (1 << n) - 1
    else:
        return -1 << -n

def maskbits(x: int, n:int) -> int:
    """Mask x & bitmask(n)"""
    if n >= 0:
        return x & ((1 << n) - 1)
    else:
        return x & (-1 << -n)

def is_even_for_rounding(c, exp):
    """General-purpose tiebreak used when rounding to even.
    If the significand is less than two bits,
    decide evenness based on the representation of the exponent.
    """
    if c.bit_length() > 1:
        return c & 1 == 0
    else:
        return exp & 1 == 0
