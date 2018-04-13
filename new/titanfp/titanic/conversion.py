"""Conversions between some common numeric types (float, np.floatXX) and
universal m/exp notation.
"""


import sys

from .integral import bitmask

import numpy as np
import gmpy2 as gmp


# Binary conversions are relatively simple for numpy's floating point types.
# float16 : w = 5,  p = 11
# float32 : w = 8,  p = 24
# float64 : w = 11, p = 53
# float128: unsupported, not an IEEE 754 128-bit float, possibly 80bit x87?
#           doc says this uses longdouble on the underlying system



def np_byteorder(ftype):
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


def float_to_mantissa_exp(f):
    """Converts a python or numpy float into universal m, exp representation:
    f = m * 2**e. If the float does not represent a real number (i.e. it is inf
    or NaN) this will raise an exception.
    """
    if isinstance(f, float):
        f = np.float64(f)
        w = 11
        pbits = 52
    elif isinstance(f, np.float16):
        w = 5
        pbits = 10
    elif isinstance(f, np.float32):
        w = 8
        pbits = 23
    elif isinstance(f, np.float64):
        w = 11
        pbits = 52
    else:
        raise TypeError('expected float or np.float{{16,32,64}}, got {}'.format(repr(type(f))))

    emax = (1 << (w - 1)) - 1

    bits = int.from_bytes(f.tobytes(), np_byteorder(type(f)))

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
        pbits = 52
        nbytes = 8
    elif ftype == np.float16:
        w = 5
        p = 11
        pbits = 10
        nbytes = 2
    elif ftype == np.float32:
        w = 8
        p = 24
        pbits = 23
        nbytes = 4
    elif ftype == np.float64:
        w = 11
        p = 53
        pbits = 52
        nbytes = 8
    else:
        raise TypeError('expected float or np.float{{16,32,64}}, got {}'.format(repr(ftype)))

    emax = (1 << (w - 1)) - 1
    emin = 1 - emax

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
        ((S << (w + pbits)) | (E << pbits) | C).to_bytes(nbytes, np_byteorder(ftype)),
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
    with gmp.context(
            precision=max(2, mbits, ebits),
            emin=min(-1, exp),
            emax=max(1, mbits, ebits, exp + mbits),
            trap_underflow=True,
            trap_overflow=True,
            trap_inexact=True,
            trap_invalid=True,
            trap_erange=True,
            trap_divzero=True,
            trap_expbound=True,
    ):
        c = gmp.mpfr(m)
        scale = gmp.exp2(exp)
        r = gmp.mul(c, scale)
        return gmp.mpfr(r, max(2, mbits))


# Mini format datasheet.
# For all formats:
#   emax = (1 << (w - 1)) - 1
#   emin = 1 - emax
#   n = emin - p

# numpy.float64 or float:
#   w = 11
#   p = 53
#   emax = 1023
#   emin = -1022
#   n = -1075
#   pbits = 52
#   nbytes = 8
# numpy.float32:
#   w = 8
#   p = 24
#   emax = 127
#   emin = -126
#   n = -150
#   pbits = 23
#   nbytes = 4
# numpy.float16
#   w = 5
#   p = 11
#   emax = 15
#   emin = -14
#   n = -25
#   pbits = 10
#   nbytes = 2


# Some basic tests:


# denormal numbers will puke out more precision than they really have;
# this is harmless for correctness testing, as all we're going to do
# is compare things for exact equality
def _float_to_mpfr(f):
    if isinstance(f, float):
        p = 53
        n = -1075
        emin = -1022
        emax = 1023
    elif isinstance(f, np.float16):
        p = 11
        n = -25
        emin = -14
        emax = 15
        f = float(f)
    elif isinstance(f, np.float32):
        p = 24
        n = -150
        emin = -126
        emax = 127
        f = float(f)
    elif isinstance(f, np.float64):
        p = 53
        n = -1075
        emin = -1022
        emax = 1023
        f = float(f)
    else:
        raise TypeError('expected float or np.float{{16,32,64}}, got {}'.format(repr(type(f))))

    with gmp.context(
            precision=p,
            emin=n,
            emax=emax + 1,
            trap_underflow=True,
            trap_overflow=True,
            trap_inexact=True,
            trap_invalid=True,
            trap_erange=True,
            trap_divzero=True,
            trap_expbound=True,
    ):
        return gmp.mpfr(f)


# always use precision of doubles (math.frexp is for doubles anyway)
def _mpfr_from_frexp(fraction, exponent):
    with gmp.context(
            precision=53,
            emin=-1075,
            emax=1025,
            trap_underflow=True,
            trap_overflow=True,
            trap_inexact=True,
            trap_invalid=True,
            trap_erange=True,
            trap_divzero=True,
            trap_expbound=True,
    ):
        c = gmp.mpfr(fraction)
        scale = gmp.exp2(exponent)
        return gmp.mul(c, scale)


def _check_agreement(i, ftype):
    import math

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
        f = np.frombuffer(i.to_bytes(nbytes, np_byteorder(ftype)), dtype=ftype, count=1, offset=0)[0]
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

    tests = 100000
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

    
def _test_all_fp32():
    print('Watch for output ...')
    print('-- testing all np.float32 --')
    ndots = 1000
    mod = (1<<32) // ndots
    for i in range(1 << 32):
        if i % mod == 0:
            print('.', end='', flush=True)
        _check_agreement(i, np.float32)
