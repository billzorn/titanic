"""Conversions between some common numeric types (float, np.floatXX) and
universal m/exp notation.
"""


import sys


from .integral import bitmask


# Binary conversions are relatively simple for numpy's floating point types.
# float16 : w = 5,  p = 11
# float32 : w = 8,  p = 24
# float64 : w = 11, p = 53
# float128: unsupported, not an IEEE 754 128-bit float, possibly 80bit x87?
#           doc says this uses longdouble on the underlying system
import numpy as np


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


# debugging

def _nprt(x):
    print(repr(x))
    return float_from_mantissa_exp(*float_to_mantissa_exp, type(x))

def _check_conv(i, ftype):
    if ftype == np.float16:
        nbytes = 2
    elif ftype == np.float32:
        nbytes = 4
    elif ftype == np.float64:
        nbytes = 8
    else:
        raise TypeError('expected np.float{{16,32,64}}, got {}'.format(repr(type(f))))

    try:
        f = np.frombuffer(i.to_bytes(nbytes, np_byteorder(ftype)), dtype=ftype, count=1, offset=0)[0]
        m, exp = float_to_mantissa_exp(f)
        f2 = float_from_mantissa_exp(m, exp, ftype)
        if f != f2:
            print(repr(f), repr(f2), s, e, c)
    except ValueError as e:
        if not (np.isinf(f) or np.isnan(f)):
            print(repr(f), repr(f2), s, e, c)
            print('  ' + repr(e))

def _test_np():
    import random
    print('watch for output...')
    for i in range(1 << 16):
        _check_conv(i, np.float16)
    for i in range(1 << 16):
        _check_conv(random.randint(0, 1 << 32), np.float32)
    for i in range(1 << 16):
        _check_conv(random.randint(0, 1 << 64), np.float64)
    print('...done')
