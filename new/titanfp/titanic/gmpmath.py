"""Common arithmetic operations (+-*/ sqrt log exp etc.)
implemented with GMP as a backend, but conveniently extended to Sinking Point.
"""


import gmpy2 as gmp

from .integral import bitmask
from . import conversion
from . import ops
from . import digital
from .sinking import Sink


def mpfr(x, prec):
    with gmp.context(
            # one extra bit, so that we can round from RTZ to RNE
            precision=prec + 1,
            emin=gmp.get_emin_min(),
            emax=gmp.get_emax_max(),
            trap_underflow=True,
            trap_overflow=True,
            trap_inexact=False,
            trap_invalid=True,
            trap_erange=True,
            trap_divzero=True,
            trap_expbound=True,
            # use RTZ for easy multiple rounding later
            round=gmp.RoundToZero,
    ):
        return gmp.mpfr(x)


def digital_to_mpfr(x):
    if x.isnan:
        with gmp.context(
                precision=2,
                emin=-1,
                emax=1,
                trap_underflow=True,
                trap_overflow=True,
                trap_inexact=True,
                trap_invalid=True,
                trap_erange=True,
                trap_divzero=True,
                trap_expbound=True,
        ):
            return gmp.mpfr('nan')
    elif x.isinf:
         with gmp.context(
                precision=2,
                emin=-1,
                emax=1,
                trap_underflow=True,
                trap_overflow=True,
                trap_inexact=True,
                trap_invalid=True,
                trap_erange=True,
                trap_divzero=True,
                trap_expbound=True,
        ):
             if x.negative:
                 return gmp.mpfr('-inf')
             else:
                 return gmp.mpfr('+inf')

    m = x.m
    exp = x.exp

    mbits = m.bit_length()
    ebits = exp.bit_length()

    # Apparently a multiplication between a small precision 0 and a huge
    # scale can raise a Type error indicating that gmp.mul() requires two
    # mpfr arguments - we can avoid that case entirely by special-casing
    # away the multiplication.
    if mbits == 0:
        with gmp.context(
            precision=2,
            emin=-1,
            emax=1,
            trap_underflow=True,
            trap_overflow=True,
            trap_inexact=True,
            trap_invalid=True,
            trap_erange=True,
            trap_divzero=True,
            trap_expbound=True,
        ):
            return gmp.mpfr(0)

    else:
        with gmp.context(
                precision=max(2, ebits),
                emin=min(-1, exp),
                emax=max(1, ebits, exp + 1),
                trap_underflow=True,
                trap_overflow=True,
                trap_inexact=True,
                trap_invalid=True,
                trap_erange=True,
                trap_divzero=True,
                trap_expbound=True,
        ):
            scale = gmp.exp2(exp)

        with gmp.context(
                precision=max(2, mbits),
                emin=min(-1, exp),
                emax=max(1, mbits, exp + mbits),
                trap_underflow=True,
                trap_overflow=True,
                trap_inexact=True,
                trap_invalid=True,
                trap_erange=True,
                trap_divzero=True,
                trap_expbound=True,
        ):
            c = gmp.mpfr(m)
            return gmp.mul(c, scale)


def mpfr_to_digital(x):
    if gmp.is_nan(x):
        return digital.Digital(
            isnan=True,
            rc=0, # not clear what the return code should mean
        )

    negative = gmp.is_signed(x)

    # Convert the result code. For MPFRs, 1 indicates that the approximate MPFR
    # is larger than the ideal, infinite-precision result (i.e. we rounded up)
    # and -1 indicates that the MPFR is less than the infinite-precision result.
    # We need to convert this to a different code used by titanic: the result
    # code is a tiny additional factor that we would have to add to the magnitude to
    # get the right answer, so if we rounded away from zero, it's -1, and if we rounded
    # towards zero, it's 1.

    if negative:
        rc = x.rc
    else:
        rc = -x.rc

    if gmp.is_infinite(x):
        return digital.Digital(
            negative=negative,
            isinf=True,
            rc=rc,
        )

    m, exp = x.as_mantissa_exp()
    c = int(abs(m))
    exp = int(exp)

    if c == 0 and rc == -1:
        raise ValueError('unreachable: MPFR rounded the wrong way toward zero? got {}, rc={}'
                         .format(repr(x), repr(x.rc)))

    return digital.Digital(
        negative=negative,
        c=c,
        exp=exp,
        inexact=(rc != 0),
        rc=rc,
    )


def _fdim(x1, x2):
    raise ValueError('fdim: emulated')
def _fmax(x1, x2):
    raise ValueError('fmax: emulated')
def _fmin(x1, x2):
    raise ValueError('fmin: emulated')

gmp_ops = [
    gmp.add,
    gmp.sub,
    gmp.mul,
    gmp.div,
    lambda x: -x,
    gmp.sqrt,
    gmp.fma,
    gmp.copy_sign,
    lambda x: abs(x),
    _fdim,
    _fmax,
    _fmin,
    gmp.fmod,
    gmp.remainder,
    gmp.ceil,
    gmp.floor,
    gmp.rint,
    gmp.round_away,
    gmp.trunc,
    gmp.acos,
    gmp.acosh,
    gmp.asin,
    gmp.asinh,
    gmp.atan,
    gmp.atan2,
    gmp.atanh,
    gmp.cos,
    gmp.cosh,
    gmp.sin,
    gmp.sinh,
    gmp.tan,
    gmp.tanh,
    gmp.exp,
    gmp.exp2,
    gmp.expm1,
    gmp.log,
    gmp.log10,
    gmp.log1p,
    gmp.log2,
    gmp.cbrt,
    gmp.hypot,
    lambda x1, x2: x1 ** x2,
    gmp.erf,
    gmp.erfc,
    lambda x: gmp.lgamma(x)[0],
    gmp.gamma,
]


def compute(opcode, *args, prec=53):
    """Compute op(*args), with up to prec bits of precision.
    op is specified via opcode, and arguments are universal digital numbers.
    Arguments are treated as exact: the inexactness and result code of the result
    only reflect what happened during this single operation.
    Result is truncated towards 0, but will have inexactness and result code set
    for further rounding, and it is computed with one extra bit of precision.
    NOTE: this function does not trap on invalid operations, so it will give the gmp/mpfr answer
    for special cases like sqrt(-1), arcsin(3), and so on.
    """
    op = gmp_ops[opcode]
    inputs = [digital_to_mpfr(arg) for arg in args]
    # gmpy2 really doesn't like it when you pass nan as an argument
    for f in inputs:
        if gmp.is_nan(f):
            return digital.Digital(isnan=True, rc=0)
    with gmp.context(
            # one extra bit, so that we can round from RTZ to RNE
            precision=prec + 1,
            emin=gmp.get_emin_min(),
            emax=gmp.get_emax_max(),
            subnormalize=False,
            # in theory, we'd like to know about these...
            trap_underflow=True,
            trap_overflow=True,
            # inexact and invalid operations should not be a problem
            trap_inexact=False,
            trap_invalid=False,
            trap_erange=False,
            trap_divzero=False,
            # We'd really like to know about this as well, but it causes i.e.
            #   mul(-25, inf) -> raise TypeError("mul() requires 'mpfr','mpfr' arguments")
            # I don't know if that behavior is more hilarious or annoying.
            trap_expbound=False,
            # use RTZ for easy multiple rounding later
            round=gmp.RoundToZero,
    ) as gmpctx:
        result = op(*inputs)

    return mpfr_to_digital(result)


def compute_constant(name, prec=53):
    with gmp.context(
            # one extra bit, so that we can round from RTZ to RNE
            precision=prec + 1,
            emin=gmp.get_emin_min(),
            emax=gmp.get_emax_max(),
            subnormalize=False,
            # in theory, we'd like to know about these...
            trap_underflow=True,
            trap_overflow=True,
            # inexact and invalid operations should not be a problem
            trap_inexact=False,
            trap_invalid=False,
            trap_erange=False,
            trap_divzero=False,
            # We'd really like to know about this as well, but it causes i.e.
            #   mul(-25, inf) -> raise TypeError("mul() requires 'mpfr','mpfr' arguments")
            # I don't know if that behavior is more hilarious or annoying.
            trap_expbound=False,
            # use RTZ for easy multiple rounding later
            round=gmp.RoundToZero,
    ) as gmpctx:
        if name == 'E':
            result = gmp.exp(1)
        elif name == 'PI':
            result = gmp.const_pi()
        else:
            raise ValueError('unsupported constant {}'.format(repr(exn.args[0])))

    return mpfr_to_digital(result)


def compute_digits(m, e, b, prec=53):
    """Compute m * b**e, with precision equal to prec. e and b must be integers, and
    b must be at least 2.
    """
    if (not isinstance(e, int)) or (not isinstance(b, int)) or (b < 2):
        raise ValueError('compute_digits: must have integer e, b, and b >= 2, got e={}, b={}'
                         .format(repr(e), repr(b)))

    with gmp.context(
            precision=max(e.bit_length(), b.bit_length()),
            emin=gmp.get_emin_min(),
            emax=gmp.get_emax_max(),
            trap_underflow=True,
            trap_overflow=True,
            trap_inexact=True,
            trap_invalid=True,
            trap_erange=True,
            trap_divzero=True,
            trap_expbound=True,
            round=gmp.RoundToZero,
    ) as gmpctx:
        mpfr_e = gmp.mpfr(e)
        mpfr_b = gmp.mpfr(b)

    with gmp.context(
            # this seems like it's enough extra bits, but I don't have a proof
            precision=prec + 3,
            emin=gmp.get_emin_min(),
            emax=gmp.get_emax_max(),
            trap_underflow=True,
            trap_overflow=True,
            trap_inexact=False,
            trap_invalid=True,
            trap_erange=True,
            trap_divzero=True,
            trap_expbound=True,
            # use RTZ for easy multiple rounding later
            round=gmp.RoundToZero,
    ) as gmpctx:
        mpfr_m = gmp.mpfr(m)
        scale = mpfr_b ** mpfr_e
        result = mpfr_m * scale

    return mpfr_to_digital(result)


def ieee_fbound(w, p):
    """Compute the boundary where IEEE 754 floating-point values
    will be rounded away to infinity for a given w and p.
    """
    emax = (1 << (w - 1)) - 1

    with gmp.context(
            precision=p + 1,
            emin=gmp.get_emin_min(),
            emax=gmp.get_emax_max(),
            trap_underflow=True,
            trap_overflow=True,
            trap_inexact=True,
            trap_invalid=True,
            trap_erange=True,
            trap_divzero=True,
            trap_expbound=True,
    ):
        fbound_scale = gmp.mpfr(2) - gmp.exp2(-p)
        fbound = gmp.exp2(emax) * fbound_scale

    return mpfr_to_digital(fbound)


def arith_sim(a, b):
    """Compute the 'arithmetic bit similarity' between a and b, defined as:
                  | a - b |
        -log2( --------------- )
               min( |a|, |b| )
    That is to say, arithmetic similarity is the negative log base 2 of the
    relative difference between a and b, with reference to whichever has
    smaller magnitude. For positive results, this is roughly an upper bound
    on the number of binary digits that are the same between the two numbers;
    for negative results, it is roughly the negative magnitude of the difference
    in exponents.
    """

    prec = max(53, 1 + max(a.e, b.e) - min(a.n, b.n))

    mpfr_a = digital_to_mpfr(a)
    mpfr_b = digital_to_mpfr(b)

    if gmp.is_nan(mpfr_a) or gmp.is_nan(mpfr_b):
        return float('nan')

    if mpfr_a == mpfr_b:
        return float('inf')
    
    with gmp.context(
            precision=prec,
            emin=gmp.get_emin_min(),
            emax=gmp.get_emax_max(),
            trap_underflow=True,
            trap_overflow=True,
            trap_inexact=True,
            trap_invalid=True,
            trap_erange=False,
            trap_divzero=True,
            trap_expbound=False
    ):
        diff = abs(mpfr_a - mpfr_b)

    with gmp.context(
            precision=prec,
            emin=gmp.get_emin_min(),
            emax=gmp.get_emax_max(),
            trap_underflow=True,
            trap_overflow=True,
            trap_inexact=False,
            trap_invalid=True,
            trap_erange=False,
            trap_divzero=False,
            trap_expbound=False
    ):
        reldiff = diff / min(abs(mpfr_a), abs(mpfr_b))

    with gmp.ieee(64):
        sim = -gmp.log2(reldiff)

    return float(sim)


def geo_sim(a, b):
    """Compute the 'geometric bit similarity' between a and b, defined as:
               |        a    |
        -log2( | log2( --- ) | )
               |        b    |
    That is to say, geometric similarity is the negative log base 2 of the
    magnitude of the log base 2 of the ratio a / b. For positive results, this
    is roughly an upper bound on the number of binary digits that are the same
    between the numbers; for negative results, it is roughtly the negative magnitude
    of the number of bits that are different between the exponents.

    In general, the geometric similarity is probably more useful when trying
    to interpret fractional values, though for positive results, the floors of the
    arithmetic and geometric similarities will usually agree.

    This measure is the same as John Gustafson's "decimal accuracy," as defined
    in https://posithub.org/docs/Posits4.pdf, section 7.4.
    """
    prec = max(53, 1 + max(a.e, b.e) - min(a.n, b.n))

    mpfr_a = digital_to_mpfr(a)
    mpfr_b = digital_to_mpfr(b)

    if gmp.is_nan(mpfr_a) or gmp.is_nan(mpfr_b):
        return float('nan')

    if mpfr_a == 0 and mpfr_b == 0:
        return float('inf')
    elif mpfr_a == 0 or mpfr_b == 0:
        return float('-inf')

    with gmp.context(
            precision=prec,
            emin=gmp.get_emin_min(),
            emax=gmp.get_emax_max(),
            trap_underflow=True,
            trap_overflow=True,
            trap_inexact=False,
            trap_invalid=True,
            trap_erange=True,
            trap_divzero=False,
            trap_expbound=False,
    ):
        ratio = mpfr_a / mpfr_b
        if ratio <= 0:
            return float('-inf')
        reldiff = abs(gmp.log2(ratio))

    with gmp.ieee(64):
        sim = -gmp.log2(reldiff)

    return float(sim)

def geo_sim10(a, b):
    """Compute the 'decimals of accuracy' between a and b, defined as:
                |        a    |
        -log10( | log10( --- ) | )
                |        b    |

    """
    prec = max(53, 1 + max(a.e, b.e) - min(a.n, b.n))

    mpfr_a = digital_to_mpfr(a)
    mpfr_b = digital_to_mpfr(b)

    if gmp.is_nan(mpfr_a) or gmp.is_nan(mpfr_b):
        return float('nan')

    if mpfr_a == 0 and mpfr_b == 0:
        return float('inf')
    elif mpfr_a == 0 or mpfr_b == 0:
        return float('-inf')

    with gmp.context(
            precision=prec,
            emin=gmp.get_emin_min(),
            emax=gmp.get_emax_max(),
            trap_underflow=True,
            trap_overflow=True,
            trap_inexact=False,
            trap_invalid=True,
            trap_erange=True,
            trap_divzero=False,
            trap_expbound=False,
    ):
        ratio = mpfr_a / mpfr_b
        if ratio <= 0:
            return float('-inf')
        reldiff = abs(gmp.log10(ratio))

    with gmp.ieee(64):
        sim = -gmp.log10(reldiff)

    return float(sim)


# deprecated


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

    # special case for the exponent of zero
    if x.is_zero() or y.is_zero():
        e = x.e + y.e
        return Sink(c = 0,
                    exp = e + 1, # since for 0, e = n, and n = exp - 1
                    negative = x.negative != y.negative,
                    inexact = not (x.is_exactly_zero() or y.is_exactly_zero()),
                    # TODO interval stuff
                    sided = False,
                    full = False)

    result = withnprec(gmp.mul, x.to_mpfr(), y.to_mpfr(),
                       min_n=min_n, max_p=max_p)

    inexact = x.inexact or y.inexact or result.inexact

    #TODO technically this could do clever things with the interval
    return Sink(result, inexact=inexact, full=False, sided=False)


def div(x, y, min_n = -1075, max_p = 53):
    """Divide to sinks x / y, rounding according to min_n and max_p.
    TODO: rounding modes
    """

    # special case for the exponent of zero
    if x.is_zero():
        e = x.e - y.e
        return Sink(c = 0,
                    exp = e + 1, # since for 0, e = n, and n = exp - 1
                    negative = x.negative != y.negative,
                    inexact = not x.is_exactly_zero(),
                    # TODO interval stuff
                    sided = False,
                    full = False)

    elif y.is_zero():
        raise ZeroDivisionError('division by zero: {} / {}'.format(repr(x), repr(y)))

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
