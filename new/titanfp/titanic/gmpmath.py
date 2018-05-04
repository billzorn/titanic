"""Common arithmetic operations (+-*/ sqrt log exp etc.)
implemented with GMP as a backend, but conveniently extended to Sinking Point.
"""


import gmpy2 as gmp

from .integral import bitmask
from . import conversion
from .sinking import Sink


def withnprec(op, *args, min_n = -1075, max_p = 53,
               emin = gmp.get_emin_min(), emax = gmp.get_emax_max()):
    """Compute op(*args), with n >= min_n and precision <= max_p.

    Arguments are provided as mpfrs; they are treated as exact values, so their
    precision is unimportant except where it affects their value. The result is a
    Titanic Sink.

    The requested precision can be less than gmpy2's minimum of 2 bits, and zeros
    produced by rounding should have all the right properties.

    TODO: the arguments themselves should not be zero; for computations involving
    zero, special cases should be written per operation.

    TODO: the output is correctly rounded under RNE rules; others should probably
    be supported.
    """

    if max_p < 0:
        raise ValueError('cannot compute a result with less than 0 max precision, got {}'.format(repr(max_p)))

    # The precision we want to compute with is at least p+1, to determine RNE behavior from RTZ.
    # We use max_p + 2 to ensure the quantity is at least 2 for mpfr.
    prec = max_p + 2

    # This context allows us to tolerate inexactness, but no other surprising behavior. Those
    # cases should be handled explicitly per operation.
    with gmp.context(
            precision=prec,
            emin=emin,
            emax=emax,
            trap_underflow=True,
            trap_overflow=True,
            trap_inexact=False,
            trap_invalid=True,
            trap_erange=True,
            trap_divzero=True,
            trap_expbound=True,
            # IMPORTANT: we need RTZ in order to be able to multiple-round accurately
            # to the desired precision.
            round=gmp.RoundToZero,
    ) as gmpctx:
        candidate = op(*args)
        op_inexact = gmpctx.inexact

    m, exp = conversion.mpfr_to_mantissa_exp(candidate)

    # Now we need to round to the correct number of bits

    mbits = m.bit_length()

    n = exp - 1
    e = n + mbits
    target_n = max(e - max_p, min_n)
    xbits = target_n - n

    # Split the result into 3 components: sign, significant bits (rounded down), and half bit

    negative = conversion.is_neg(candidate)
    c = abs(m)

    if c > 0:
        sig = c >> xbits
        half_x = c & bitmask(xbits)
        half = half_x >> (xbits - 1)
        x = half_x & bitmask(xbits - 1)
    else:
        sig = 0
        half_x = 0
        half = 0
        x = 0

    # Now we need to decide how to round. The value we have in sig was rounded toward zero, so we
    # look at the half bit and the inexactness of the operation to decide if we should round away.

    if half > 0:
        # greater than halfway away implied by inexactness, or demonstrated by nonzero xbits
        if x > 0 or op_inexact:
            # if we have no precision, round away by increasing n
            if max_p == 0:
                target_n += 1
            else:
                sig += 1
        # TODO: hardcoded RNE
        elif sig & 1 > 0:
            sig += 1

    # fixup extra precision from a carry out
    # TODO in theory this could all be abstracted away with .away()
    if sig.bit_length() > max_p:
        # TODO assert
        if sig & 1 > 0:
            raise AssertionError('cannot fixup extra precision: {} {}'.format(repr(op), repr(args))
                                 + '  {}, m={}, exp={}, decoded as {} {} {} {}'.format(
                                     repr(candidate), repr(m), repr(exp), negative, sig, half, x))
        sig >>= 1
        target_n += 1

    result_inexact = half > 0 or x > 0 or op_inexact
    result_sided = sig == 0 and not result_inexact

    return Sink(c=sig,
                exp=target_n + 1,
                negative=negative,
                inexact=result_inexact,
                sided=result_sided,
                full=False)


# All these operations proceed exactly with the bits that they are given,
# which means that they can produce results that are more precise than should
# be allowed given the inexactness of their inputs. The idea is not to call them
# with unacceptably precise rounding specifications.

# These operations won't crash if given inexact zeros, but nor will they do anything
# clever with the precision / exponent of the result.

# For zeros that are computed from non-zero inputs, the precision / exponent
# should be reasonable.

# Though extra precision may be present, the inexact flags of the results should
# always be set correctly.

# Actually, do these wrappers actually do anything? In all cases it seems like
# we just want a computed zero to have the specified min_n...

def add(x, y, min_n = -1075, max_p = 53):
    """Add two sinks, rounding according to min_n and max_p.
    TODO: rounding modes
    """
    result = withnprec(gmp.add, x.to_mpfr(), y.to_mpfr(),
                       min_n=min_n, max_p=max_p)

    inexact = x.inexact or y.inexact or result.inexact

    #TODO technically this could do clever things with the interval
    return Sink(result, inexact=inexact, full=False, sided=False)


def sub(x, y, min_n = -1075, max_p = 53):
    """Subtract two sinks, rounding according to min_n and max_p.
    TODO: rounding modes
    """
    result = withnprec(gmp.sub, x.to_mpfr(), y.to_mpfr(),
                       min_n=min_n, max_p=max_p)

    inexact = x.inexact or y.inexact or result.inexact

    #TODO technically this could do clever things with the interval
    return Sink(result, inexact=inexact, full=False, sided=False)


def mul(x, y, min_n = -1075, max_p = 53):
    """Multiply two sinks, rounding according to min_n and max_p.
    TODO: rounding modes
    """
    result = withnprec(gmp.mul, x.to_mpfr(), y.to_mpfr(),
                       min_n=min_n, max_p=max_p)

    inexact = x.inexact or y.inexact or result.inexact

    #TODO technically this could do clever things with the interval
    return Sink(result, inexact=inexact, full=False, sided=False)


def div(x, y, min_n = -1075, max_p = 53):
    """Divide to sinks x / y, rounding according to min_n and max_p.
    TODO: rounding modes
    """
    result = withnprec(gmp.div, x.to_mpfr(), y.to_mpfr(),
                       min_n=min_n, max_p=max_p)

    inexact = x.inexact or y.inexact or result.inexact

    #TODO technically this could do clever things with the interval
    return Sink(result, inexact=inexact, full=False, sided=False)


def sqrt(x, min_n = -1075, max_p = 53):
    """Take the square root of a sink, rounding according to min_n and max_p.
    TODO: rounding modes
    """
    result = withnprec(gmp.sqrt, x.to_mpfr(),
                       min_n=min_n, max_p=max_p)

    inexact = x.inexact or result.inexact

    #TODO technically this could do clever things with the interval
    return Sink(result, inexact=inexact, full=False, sided=False)


def floor(x, min_n = -1075, max_p = 53):
    """Take the floor of x, rounding according to min_n and max_p.
    TODO: rounding modes
    """
    result = withnprec(gmp.floor, x.to_mpfr(),
                       min_n=min_n, max_p=max_p)

    # TODO it's noot 100% clear who should do the checking here.
    # Technically, any exactly computed floor is exact unless
    # its n permits an ulp variance of at least unity.

    # However, implementations may want to decide that certain floors
    # are inexact even though the representation does not require them
    # to be so, i.e. floor(1.00000000000~) could be 0 or 1 depending
    # on which way we rounded, even though at that precision ulps are
    # relatively small.

    inexact = (x.inexact or result.inexact) and min_n >= 0

    #TODO technically this could do clever things with the interval
    return Sink(result, inexact=inexact, full=False, sided=False)


def fmod(x, y, min_n = -1075, max_p = 53):
    """Compute the remainder of x mod y, rounding according to min_n and max_p.
    TODO: rounding modes
    """
    result = withnprec(gmp.fmod, x.to_mpfr(), y.to_mpfr(),
                       min_n=min_n, max_p=max_p)

    inexact = x.inexact or y.inexact or result.inexact

    #TODO technically this could do clever things with the interval
    return Sink(result, inexact=inexact, full=False, sided=False)


def pow(x, y, min_n = -1075, max_p = 53):
    """Raise x ** y, rounding according to min_n and max_p.
    TODO: rounding modes
    """
    result = withnprec(lambda x, y: x**y, x.to_mpfr(), y.to_mpfr(),
                       min_n=min_n, max_p=max_p)

    inexact = x.inexact or y.inexact or result.inexact

    #TODO technically this could do clever things with the interval
    return Sink(result, inexact=inexact, full=False, sided=False)


def sin(x, min_n = -1075, max_p = 53):
    """Compute sin(x), rounding according to min_n and max_p.
    TODO: rounding modes
    """
    result = withnprec(gmp.sin, x.to_mpfr(),
                       min_n=min_n, max_p=max_p)

    inexact = x.inexact or result.inexact

    #TODO technically this could do clever things with the interval
    return Sink(result, inexact=inexact, full=False, sided=False)


# helpers to produce some useful constants

def pi(p):
    # TODO no support for rounding modes
    if p < 0:
        raise ValueError('precision must be at least 0')
    elif p == 0:
        # TODO is this right???
        return Sink(m=0, exp=3, inexact=True, sided=True, full=False)
    elif p == 1:
        return Sink(m=1, exp=2, inexact=True, sided=False, full=False)
    else:
        with gmp.context(
                    precision=p,
                    emin=-1,
                    emax=2,
                    trap_underflow=True,
                    trap_overflow=True,
                    trap_inexact=False,
                    trap_invalid=True,
                    trap_erange=True,
                    trap_divzero=True,
                    trap_expbound=True,
        ):
            x = gmp.const_pi()
    return Sink(x, inexact=True, sided=False, full=False)

def mpfr(x, p):
    # TODO no support for rounding modes
    if p < 2:
        raise ValueError('precision must be at least 2')
    else:
        with gmp.context(
                    precision=p,
                    emin=gmp.get_emin_min(),
                    emax=gmp.get_emax_max(),
                    trap_underflow=True,
                    trap_overflow=True,
                    trap_inexact=False,
                    trap_invalid=True,
                    trap_erange=True,
                    trap_divzero=True,
                    trap_expbound=True,
        ):
            r = gmp.mpfr(x)
        return r
