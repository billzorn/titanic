"""Common arithmetic operations (+-*/ sqrt log exp etc.)
implemented with GMP as a backend, but conveniently extended to Sinking Point.
"""


import gmpy2 as gmp

from .integral import bitmask
from .conversion import mpfr_to_mantissa_exp
from .sinking import Sink


_DEFAULT_XULPS = 0
_DEFAULT_XBITS_MAX = 1 << 20
_DEFAULT_EMIN = -(1 << 60)
_DEFAULT_EMAX = 1 << 60


_TODO_FIXME_ZERO = Sink(0, n=0)


def withnprec(op, *args, min_n = -1075, max_p = 53, xulps = _DEFAULT_XULPS,
               emin = _DEFAULT_EMIN, emax = _DEFAULT_EMAX):
    """Compute op(*args), with n >= min_n and precision <= max_p.

    TODO: exactness of input sinks -> exactness of result

    Arguments are provided as mpfrs; they are treated as exact values, so their
    precision is unimportant except where it affects their value. The result is a
    Titanic Sink.

    The requested precision can be less than gmpy2's minimum of 2 bits, and zeros
    produced by rounding should have all the right properties.

    TODO: the arguments themselves should not be zero; for computations involving
    zero, special cases should be written per operation.

    If xulps is greater than zero, it specifies the minimum number of ulps an
    inexact answer must be away from a decision boundary to allow rounding to
    proceed. This way, the algorithm is robust to an underlying gmp implementation
    that is within xulps of the correctly rounded answer. If the margin is too small,
    the answer is recomputed with a geometrically increasing amount of precision, until
    either an exact answer is produced, or the margin is large enough.

    TODO: the output is correctly rounded under RNE rules; others should probably
    be supported.
    """

    # or should the arguments be sinks
    args = [arg.to_mpfr() for arg in args]
    
    xbits = xulps.bit_length() << 1

    while xbits <= _DEFAULT_XBITS_MAX:

        # the precision we want to use is p, +1 to determine RNE behavior from RTZ, +xbits
        prec = max(2, max_p + 1 + xbits)

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

        m, exp = mpfr_to_mantissa_exp(candidate)

        # TODO assert
        if m == 0:
            raise AssertionError('unexpected zero in computation: {} {}'.format(repr(op), repr(args)))

        # We have to do two things at this point:
        #  Determine if the result is outside the margin of xulps,
        #  and round to the correct number of bits if it is.

        mbits = m.bit_length()

        n = exp - 1
        e = n + mbits
        target_n = max(e - max_p, min_n)

        computed_xbits = target_n - n - 1
        # TODO assert
        if computed_xbits < 0:
            raise AssertionError('operation returned insufficient precision: {} {}'.format(repr(op), repr(args))
                                 + '  got {} with n={}, expecting n < {}'.format(repr(candidate), n, target_n))

        # Split the result into 4 components: sign, significant bits (rounded down), half bit, xbits

        if m >= 0:
            negative = False
            c = m
        else:
            negative = True
            c = -m

        sig = c >> (computed_xbits + 1)
        half = (c >> computed_xbits) & 1
        x = c & bitmask(computed_xbits)

        # If we're too close to the boundary, try again with more xbits
        if op_inexact and xulps > 0 and not xulps <= x <= (1 << computed_xbits) - xulps:
            xbits <<= 1
            continue

        # Now we need to decide how to round. The value we have in sig was rounded toward zero, so we
        # look at the half bit, the xbits, and the inexactness of the operation to decide if we should
        # round away.

        if half > 0:
            # greater than halfway away, either by direct proof, or implied by inexactness
            if x > 0 or op_inexact:
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

        return Sink(x = _TODO_FIXME_ZERO,
                    e = target_n + sig.bit_length(),
                    n = target_n,
                    p = sig.bit_length(),
                    c = sig,
                    negative = negative,
                    inexact = result_inexact,
                    sided_interval = result_sided,
                    full_interval = False)

def pi(p):
    assert p >= 2
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
        return gmp.const_pi()
