"""Conversions between some common numeric types (float, np.floatXX) and
universal m/exp notation.
"""


import sys
import math

import numpy as np
import gmpy2 as gmp
mpfr_t = type(gmp.mpfr())

from .integral import bitmask


# reusable definitions
ROUND_NEAREST_EVEN = 1
ROUND_NEAREST_AWAY = 2
ROUND_UP = 3
ROUND_DOWN = 4
ROUND_TO_ZERO = 5
ROUND_AWAY_FROM_ZERO = 6

def _np_byteorder(ftype):
    """Converts from numpy byteorder conventions for a floating point datatype
    to sys.byteorder 'big' or 'little'.
    """
    bo = np.dtype(ftype).byteorder
    if bo == '=':
        return sys.byteorder
    elif bo == '<':
        return 'little'
    elif bo == '>':
        return 'big'
    else:
        raise ValueError('unknown numpy byteorder {} for dtype {}'.format(repr(bo), repr(ftype)))


# Mini format datasheet.
# For all formats:
#   emax = (1 << (w - 1)) - 1
#   emin = 1 - emax
#   n = emin - p

def fdata(ftype):
    if ftype == np.float16:
        return {
            'w' : 5,
            'p' : 11,
            'emax' : 15,
            'emin' : -14,
            'n' : -25,
            'pbits' : 10,
            'nbytes' : 2,
        }
    elif ftype == np.float32:
        return {
            'w' : 8,
            'p' : 24,
            'emax' : 127,
            'emin' : -126,
            'n' : -150,
            'pbits' : 23,
            'nbytes' : 4,
        }
    elif ftype == np.float64 or ftype == float:
        return {
            'w' : 11,
            'p' : 53,
            'emax' : 1023,
            'emin' : -1022,
            'n' : -1075,
            'pbits' : 52,
            'nbytes' : 8,
        }
    elif ftype == np.float128:
        raise ValueError('unsupported: machine-dependent 80-bit longdouble')
    else:
        TypeError('expected float or np.float{{16,32,64}}, got {}'.format(repr(ftype)))


# Get status flags out of numeric types

def is_neg(f):
    """Get the sign bit of a float or mpfr."""
    if isinstance(f, mpfr_t):
        if gmp.is_zero(f):
            #TODO: this is terrible
            return str(f).startswith('-')
        elif gmp.is_nan(f):
            raise ValueError('mpfr NaNs are unsigned')
        else:
            return gmp.sign(f) < 0
    else:
        if isinstance(f, float):
            f = np.float64(f)
            offset = 63
        elif isinstance(f, np.float16):
            offset = 15
        elif isinstance(f, np.float32):
            offset = 31
        elif isinstance(f, np.float64):
            offset = 63
        else:
            raise TypeError('expected mpfr, float, or np.float{{16,32,64}}, got {}'.format(repr(type(f))))

        bits = int.from_bytes(f.tobytes(), _np_byteorder(type(f)))

        return bits >> offset != 0

def is_inf(f):
    """Check whether a float or mpfr is inf or -inf."""
    if isinstance(f, mpfr_t):
        return gmp.is_inf(f)
    elif isinstance(f, float):
        return math.isinf(f)
    elif isinstance(f, np.float16):
        return np.isinf(f)
    elif isinstance(f, np.float32):
        return np.isinf(f)
    elif isinstance(f, np.float64):
        return np.isinf(f)
    else:
        raise TypeError('expected mpfr, float, or np.float{{16,32,64}}, got {}'.format(repr(type(f))))

def is_nan(f):
    """Check whether a float or mpfr is nan."""
    if isinstance(f, mpfr_t):
        return gmp.is_nan(f)
    elif isinstance(f, float):
        return math.isnan(f)
    elif isinstance(f, np.float16):
        return np.isnan(f)
    elif isinstance(f, np.float32):
        return np.isnan(f)
    elif isinstance(f, np.float64):
        return np.isnan(f)
    else:
        raise TypeError('expected mpfr, float, or np.float{{16,32,64}}, got {}'.format(repr(type(f))))


# Other non-real properties of numeric types

def float_to_payload(f):
    """Get the integer payload of a float that is NaN."""
    if isinstance(f, float):
        pbits = 52
        f = np.float64(f)
    elif isinstance(f, np.float16):
        pbits = 10
    elif isinstance(f, np.float32):
        pbits = 23
    elif isinstance(f, np.float64):
        pbits = 52
    else:
        raise TypeError('expected float or np.float{{16,32,64}}, got {}'.format(repr(type(f))))

    if not np.isnan(f):
        raise ValueError('expecting NaN, got {}'.format(repr(f)))

    bits = int.from_bytes(f.tobytes(), _np_byteorder(type(f)))

    return bits & bitmask(pbits)


# Conversions to and from mantissa / exponent form


def float_to_mantissa_exp(f):
    """Converts a python or numpy float into universal m, exp representation:
    f = m * 2**e. If the float does not represent a real number (i.e. it is inf
    or NaN) this will raise an exception.
    """
    if isinstance(f, float):
        w = 11
        emax = 1023
        pbits = 52
        f = np.float64(f)
    elif isinstance(f, np.float16):
        w = 5
        emax = 15
        pbits = 10
    elif isinstance(f, np.float32):
        w = 8
        emax = 127
        pbits = 23
    elif isinstance(f, np.float64):
        w = 11
        emax = 1023
        pbits = 52
    else:
        raise TypeError('expected float or np.float{{16,32,64}}, got {}'.format(repr(type(f))))

    bits = int.from_bytes(f.tobytes(), _np_byteorder(type(f)))

    S = bits >> (w + pbits) & bitmask(1)
    E = bits >> (pbits) & bitmask(w)
    C = bits & bitmask(pbits)

    e = E - emax

    if E == 0:
        # subnormal
        if S == 0:
            m = C
        else:
            m = -C
        exp = -emax - pbits + 1
    elif e <= emax:
        # normal
        if S == 0:
             m = C | (1 << pbits)
        else:
            m = -(C | (1 << pbits))
        exp = e - pbits
    else:
        # nonreal
        raise ValueError('nonfinite value {}'.format(repr(f)))

    return m, exp


def float_from_mantissa_exp(m, exp, ftype=float):
    """Converts universal m, exp representation into a python or numpy
    float according to ftype.
    TODO: this implementation is incapable of rounding: if it is not given
    enough precision, it will complain to stdout, and if it is given too much,
    it will raise an exception rather than trying to round.
    """
    if ftype == float:
        w = 11
        p = 53
        emax = 1023
        emin = -1022
        pbits = 52
        nbytes = 8
    elif ftype == np.float16:
        w = 5
        p = 11
        emax = 15
        emin = -14
        pbits = 10
        nbytes = 2
    elif ftype == np.float32:
        w = 8
        p = 24
        emax = 127
        emin = -126
        pbits = 23
        nbytes = 4
    elif ftype == np.float64:
        w = 11
        p = 53
        emax = 1023
        emin = -1022
        pbits = 52
        nbytes = 8
    else:
        raise TypeError('expected float or np.float{{16,32,64}}, got {}'.format(repr(ftype)))

    if m >= 0:
        S = 0
        c = m
    else:
        S = 1
        c = -m

    cbits = c.bit_length()

    e = exp + cbits - 1

    if e < emin:
        # subnormal
        lz = (emin - 1) - e
        if lz > pbits or (lz == pbits and cbits > 0):
            raise ValueError('exponent out of range: {}'.format(e))
        elif lz + cbits > pbits:
            raise ValueError('too much precision: given {}, can represent {}'.format(cbits, pbits - lz))
        E = 0
        C = c << (lz - (pbits - cbits))
    elif e <= emax:
        # normal
        if cbits > p:
            raise ValueError('too much precision: given {}, can represent {}'.format(cbits, p))
        elif cbits < p:
            print('Warning: inventing {} low order bits!'.format(p - cbits))
        E = e + emax
        C = (c << (p - cbits)) & bitmask(pbits)
    else:
        # overflow
        raise ValueError('exponent out of range: {}'.format(e))

    f = np.frombuffer(
        ((S << (w + pbits)) | (E << pbits) | C).to_bytes(nbytes, _np_byteorder(ftype)),
        dtype=ftype, count=1, offset=0,
    )[0]

    if ftype == float:
        return float(f)
    else:
        return f


def float64_from_mantissa_exp(m, exp):
    """Converts universal m, exp representation into a numpy float64,
    as float_from_mantissa_exp.
    """
    return float_from_mantissa_exp(m, exp, ftype=np.float64)


def float32_from_mantissa_exp(m, exp):
    """Converts universal m, exp representation into a numpy float32,
    as float_from_mantissa_exp.
    """
    return float_from_mantissa_exp(m, exp, ftype=np.float32)


def float16_from_mantissa_exp(m, exp):
    """Converts universal m, exp representation into a numpy float16,
    as float_from_mantissa_exp.
    """
    return float_from_mantissa_exp(m, exp, ftype=np.float16)


def mpfr_to_mantissa_exp(f):
    """Converts a gmpy2 mpfr into universal m, exp representation:
    f = m * 2**e. If the mpfr does not represent a real number, then
    this will raise an exception. Note that for real numbers, the behavior
    is identical to f.as_mantissa_exp() except the results are converted
    to python ints instead of gmpy2.mpz.
    """
    m, exp = f.as_mantissa_exp()
    return int(m), int(exp)


def mpfr_from_mantissa_exp(m, exp):
    """Converts universal m, exp representation into a gmpy2 mpfr. The mpfr will
    always reflect the inherent precision of m and exp, unless m is fewer than 2 bits
    long, in which case the resulting mpfr will have a precision of 2.
    """
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
        ):
            c = gmp.mpfr(m)
            return gmp.mul(c, scale)


def numeric_to_mantissa_exp(x):
    """Convert any type that can be interpreted as a number to
    universal m, exp representation: x = m * 2**e.
    """
    if isinstance(x, int):
        return (x, 0)
    elif isinstance(x, float) or isinstance(x, np.float16) or isinstance(x, np.float32) or isinstance(x, np.float64):
        return float_to_mantissa_exp(x)
    elif isinstance(x, mpfr_t):
        return mpfr_to_mantissa_exp(x)
    else:
        raise TypeError('{}: not a numeric type'.format(repr(x)))


def numeric_to_signed_mantissa_exp(x):
    """Convert any type that can be interpreted as a number to
    universal sign, m, exp representation: x = m * 2**e, with an explicit
    sign so that the sign of floating point zeros is not lost.
    """
    if isinstance(x, int):
        return (x < 0, abs(x), 0)
    elif isinstance(x, float) or isinstance(x, np.float16) or isinstance(x, np.float32) or isinstance(x, np.float64):
        m, exp = float_to_mantissa_exp(x)
        return (is_neg(x), abs(m), exp)
    elif isinstance(x, mpfr_t):
        m, exp = mpfr_to_mantissa_exp(x)
        return (is_neg(x), abs(m), exp)
    else:
        raise TypeError('{}: not a numeric type'.format(repr(x)))


# Some basic tests


# denormal numbers will puke out more precision than they really have;
# this is harmless for correctness testing, as all we're going to do
# is compare things for exact equality
def _float_to_mpfr(f):
    if isinstance(f, float):
        p = 53
        n = -1075
        emin = -1022
        emax = 1024 # emax + 1
    elif isinstance(f, np.float16):
        p = 11
        n = -25
        emin = -14
        emax = 16 # emax + 1
        f = float(f)
    elif isinstance(f, np.float32):
        p = 24
        n = -150
        emin = -126
        emax = 128 # emax + 1
        f = float(f)
    elif isinstance(f, np.float64):
        p = 53
        n = -1075
        emin = -1022
        emax = 1024 # emax + 1
        f = float(f)
    else:
        raise TypeError('expected float or np.float{{16,32,64}}, got {}'.format(repr(type(f))))

    with gmp.context(
            precision=p,
            emin=n,
            emax=emax,
            trap_underflow=True,
            trap_overflow=True,
            trap_inexact=True,
            trap_invalid=True,
            trap_erange=True,
            trap_divzero=True,
    ):
        return gmp.mpfr(f)


# always use precision of doubles (math.frexp is for doubles anyway)
def _mpfr_from_frexp(fraction, exponent):
    with gmp.context(
            precision=53,
            emin=-1075,
            emax=1025, # emax + 2
            trap_underflow=True,
            trap_overflow=True,
            trap_inexact=True,
            trap_invalid=True,
            trap_erange=True,
            trap_divzero=True,
    ):
        c = gmp.mpfr(fraction)
        scale = gmp.exp2(exponent)
        return gmp.mul(c, scale)


def _check_agreement(i, ftype):
    if ftype == float:
        nbytes = 8
    elif ftype == np.float16:
        nbytes = 2
    elif ftype == np.float32:
        nbytes = 4
    elif ftype == np.float64:
        nbytes = 8
    else:
        raise TypeError('expected float or np.float{{16,32,64}}, got {}'.format(repr(type(f))))

    try:
        f = np.frombuffer(i.to_bytes(nbytes, _np_byteorder(ftype)), dtype=ftype, count=1, offset=0)[0]
        if np.isinf(f) or np.isnan(f):
            # raise OverflowError('not a real number: {}'.format(f))
            return # TODO: this could do something
        if ftype == float:
            f = float(f)
        m, exp = float_to_mantissa_exp(f)

        f1 = float_from_mantissa_exp(m, exp, ftype)

        r = mpfr_from_mantissa_exp(m, exp)
        r1 = _float_to_mpfr(f)
        r2 = _mpfr_from_frexp(*math.frexp(f))

        m1, exp1 = mpfr_to_mantissa_exp(r)

        errs = ''
        if not (f == f1):
            errs += '  f failed to round trip through m, exp: {} = {}'.format(f, f1)
        if not (m == m1 and exp == exp1):
            if m == 0:
                pass # IEEE 754 floats report small exponents for 0 (smaller than smallest denorm...), while mpfr reports 1
            elif abs(m) == 1 and abs(m1) == 2 and exp1 == exp - 1:
                pass # The smallest denorm has less than 2 precision, which mpfr can't represent
            else:
                errs += '  m, exp failed to round trip through mpfr: {} = {}, {} = {}\n'.format(m, m1, exp, exp1)
        if not r == r1 == r2:
            errs += '  mpfr forms disagree: {} = {} = {}\n'.format(repr(r), repr(r1), repr(r2))

        if errs != '':
            print('disagreement on {}, {}, {}'.format(i, ftype, f))
            print(errs)
    except Exception as e:
        print('Unexpected exception on {}, {}'.format(i, ftype))
        raise e


def _test():
    import random

    print('Watch for output ...')
    print('-- testing all np.float16 --')
    for i in range(1 << 16):
        _check_agreement(i, np.float16)

    tests = 1 << 20
    n32 = tests
    n64 = tests
    nfloat = tests

    print('-- testing {} np.float32 --'.format(n32))
    imax = bitmask(32)
    for i in range(n32):
        _check_agreement(random.randint(0, imax), np.float32)

    print('-- testing {} np.float64 --'.format(n64))
    imax = bitmask(64)
    for i in range(n64):
        _check_agreement(random.randint(0, imax), np.float64)

    print('-- testing {} float --'.format(nfloat))
    imax = bitmask(64)
    for i in range(nfloat):
        _check_agreement(random.randint(0, imax), np.float64)

    print('... Done.')


# this consumes many cores and takes cpu-days
def _test_all_fp32(ncores = None):
    import multiprocessing

    if ncores is None:
        ncores = multiprocessing.cpu_count()

    tests = 1 << 32
    blocksize = tests // ncores

    procs = []
    for n in range(ncores):
        start = n * blocksize
        if n < ncores - 1:
            end = (n+1) * blocksize
        else:
            end = tests

        p = multiprocessing.Process(target=_test_all_fp32_between, args=(start,end))
        procs.append(p)

    print('Watch for output ...')

    for p in procs:
        p.start()

    for p in procs:
        p.join()

    print('... Done.')


def _test_all_fp32_between(start, end):
    print('-- testing all np.float32 from {} to {} --'.format(start, end))

    ndots = 100 # per core
    mod = (end - start) // ndots
    output_on = mod - 1

    for i in range(start, end):
        _check_agreement(i, np.float32)
        if i % mod == output_on:
            print('.', end='', flush=True)


def _test_big_mpfrs():
    import random

    print('Watch for output ...')

    tests = 1 << 20

    print('-- testing {} big mpfrs --'.format(tests))
    for _ in range(tests):

        l = random.randint(0, 1)
        if l == 0:
            llen = random.randint(1, 1024)
            lbits = random.randint(0, bitmask(llen))
        else:
            lbits = 1

        m = random.randint(0, 1)
        if m == 0:
            offset = random.randint(1, 2048)
        else:
            offset = 0

        r = random.randint(0, 2)
        if r == 0:
            rlen = random.randint(1, 1024)
            rbits = random.randint(0, bitmask(rlen))
        elif r == 1:
            rbits = 1
        else:
            rbits = 0

        m = (lbits << (offset + rbits.bit_length())) | rbits
        exp = random.randint(-65535, 65535)

        f = mpfr_from_mantissa_exp(m, exp)

        m1, exp1 = mpfr_to_mantissa_exp(f)


        if not (m == m1 and exp == exp1):
            if m == 0:
                pass # IEEE 754 floats report small exponents for 0 (smaller than smallest denorm...), while mpfr reports 1
            elif abs(m) == 1 and abs(m1) == 2 and exp1 == exp - 1:
                pass # The smallest denorm has less than 2 precision, which mpfr can't represent
            else:
                print('disagreement on {} << {} . {}'.format(lbits, offset, rbits))
                print('  m, exp failed to round trip through mpfr: {} = {}, {} = {}\n'.format(m, m1, exp, exp1))

    print('... Done.')
