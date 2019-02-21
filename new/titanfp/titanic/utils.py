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
