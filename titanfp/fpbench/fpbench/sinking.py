"""Sinking point -
A discrete approximation of real numbers with explicit significance tracking.
Implemented really badly in one file.
"""


import sys
import random


def bitmask(n):
    if n > 0:
        return (1 << n) - 1
    else:
        return (-1) << (-n)


# Binary conversions are relatively simple for numpy's floating point types.
# float16 : w = 5,  p = 11
# float32 : w = 8,  p = 24
# float64 : w = 11, p = 53
# float128: unsupported, not an IEEE 754 128-bit float, possibly 80bit x87?
#           doc says this uses longdouble on the underlying system
import numpy as np


def np_byteorder(ftype):
    bo = np.dtype(ftype).byteorder
    if bo == '=':
        return sys.byteorder
    elif bo == '<':
        return 'little'
    elif bo == '>':
        return 'big'
    else:
        raise ValueError('unknown numpy byteorder {} for dtype {}'.format(repr(bo), repr(ftype)))


def xfloat(f):
    if isinstance(f, np.float16):
        w = 5
        pbits = 10
    elif isinstance(f, np.float32):
        w = 8
        pbits = 23
    elif isinstance(f, np.float64):
        w = 11
        pbits = 52
    else:
        raise TypeError('expected np.float{{16,32,64}}, got {}'.format(repr(type(f))))

    emax = (1 << (w - 1)) - 1

    bits = int.from_bytes(f.tobytes(), np_byteorder(type(f)))

    S = bits >> (w + pbits) & bitmask(1)
    E = bits >> (pbits) & bitmask(w)
    C = bits & bitmask(pbits)

    e = E - emax

    if E == 0:
        # subnormal
        return S != 0, -emax - (pbits - C.bit_length()), C
    elif e <= emax:
        # normal
        return S != 0, e, C | (1 << pbits)
    else:
        # nonreal
        raise ValueError('nonfinite value {}'.format(repr(f)))


def mkfloat(s, e, c, ftype=np.float64):
    if ftype == np.float16:
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
        raise TypeError('expected np.float{{16,32,64}}, got {}'.format(repr(type(f))))

    emax = (1 << (w - 1)) - 1
    emin = 1 - emax

    cbits = c.bit_length()

    if e < emin:
        # subnormal
        lz = (emin - 1) - e
        if lz > pbits or (lz == pbits and cbits > 0):
            raise ValueError('exponent out of range: {}'.format(e))
        elif lz + cbits > pbits:
            raise ValueError('too much precision: given {}, can represent {}'.format(cbits, pbits - lz))
        S = 1 if s else 0
        E = 0
        C = c << (lz - (pbits - cbits))
    elif e <= emax:
        # normal
        if cbits > p:
            raise ValueError('too much precision: given {}, can represent {}'.format(cbits, p))
        elif cbits < p:
            print('Warning: inventing {} low order bits!'.format(p - cbits))
        S = 1 if s else 0
        E = e + emax
        C = (c << (p - cbits)) & bitmask(pbits)
    else:
        # overflow
        raise ValueError('exponent out of range: {}'.format(e))

    return np.frombuffer(
        ((S << (w + pbits)) | (E << pbits) | C).to_bytes(nbytes, np_byteorder(ftype)),
        dtype=ftype, count=1, offset=0,
    )[0]


# debugging

def _nprt(x):
    print(repr(x))
    return mkfloat(*xfloat(x), type(x))

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
        s, e, c = xfloat(f)
        f2 = mkfloat(s, e, c, ftype)
        if f != f2:
            print(repr(f), repr(f2), s, e, c)
    except ValueError as e:
        if not (np.isinf(f) or np.isnan(f)):
            print(repr(f), repr(f2), s, e, c)
            print('  ' + repr(e))

def _test_np():
    print('watch for output...')
    for i in range(1 << 16):
        _check_conv(i, np.float16)
    for i in range(1 << 16):
        _check_conv(random.randint(0, 1 << 32), np.float32)
    for i in range(1 << 16):
        _check_conv(random.randint(0, 1 << 64), np.float64)
    print('...done')


# gmpy2 helpers
import gmpy2 as gmp
mpfr = gmp.mpfr
mpz = gmp.mpz


# this should probably be replaced with a better mechanism...
_GMPCTX = gmp.context()
_GMPCTX.precision = 2
_GMPCTX.trap_underflow = True
_GMPCTX.trap_overflow = True
_GMPCTX.trap_inexact = True
_GMPCTX.trap_invalid = True
_GMPCTX.trap_erange = True
_GMPCTX.trap_divzero = True
_GMPCTX.trap_expbound = True
gmp.set_context(_GMPCTX)


def sigbits(m):
    z = mpz(m)
    trailing_zeros = z.bit_scan1(0)
    if trailing_zeros is None:
        return 0
    else:
        return z.bit_length() - trailing_zeros


def to_mantissa_exp(r):
    m, e = r.as_mantissa_exp()
    trailing_zeros = m.bit_scan1(0)
    if trailing_zeros is None:
        return 0, None
    else:
        return m >> trailing_zeros, e + trailing_zeros


def from_mantissa_exp(m, e):
    emax = int(e) + m.bit_length()

    with _GMPCTX as gmpctx:
        if e <= gmpctx.emin:
            gmpctx.emin = int(e) - 1
        elif emax >= gmpctx.emax:
            gmpctx.emax = emax + 1
        gmpctx.precision = max(2, sigbits(m))

        return gmp.mul(mpfr(m), gmp.exp2(mpfr(e, max(2, sigbits(e)))))


# more debugging

def _gmprt(x1, x2=None):
    if x2 is None:
        x = mpfr(x1, 53)
    else:
        x = from_mantissa_exp(x1, x2)
    result = from_mantissa_exp(*to_mantissa_exp(x))
    print(repr(x))
    return(x)

def _test_gmp():
    print('watch for output...')
    for i in range(1 << 16):
        m = random.randint(0, 1 << 256) << (random.randint(0, 64) if random.randint(0, 2) else 0)
        e = random.randint(-126, 127) if random.randint(0, 2) else random.randint(-(1 << 32), (1 << 32))
        x1 = from_mantissa_exp(m, e)
        x2 = from_mantissa_exp(*to_mantissa_exp(x1))
        if x1 != x2:
            print(m, e, x1, x2)
    print('...done')


class Sink:
    e : int = None # exponent
    n : int = None # "sticky bit" or lsb
    p : int = None # precision: e - n
    c : int = None # significand
    negative : bool = None # sign bit
    inexact : bool = None # approximate bit
    inf: bool = None # is the value infinite?
    nan: bool = None # is this value NaN?

    def _valid(self) -> bool:
        return (
            (self.e >= self.n) and
            (self.p == self.e - self.n) and
            (not self.inf and self.nan)
        )

    def __init__(self, x) -> None:
        pass
