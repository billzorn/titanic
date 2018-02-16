"""Sinking point -
A discrete approximation of real numbers with explicit significance tracking.
Implemented really badly in one file.
"""


import typing
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


def exactctx(prec, emin, emax):
    return gmp.context(
        precision=max(2, prec),
        emin=min(-1, emin),
        emax=max(+1, emax),
        trap_underflow=True,
        trap_overflow=True,
        trap_inexact=True,
        trap_invalid=True,
        trap_erange=True,
        trap_divzero=True,
        trap_expbound=True,
    )


def sigbits(m):
    z = mpz(m)
    trailing_zeros = z.bit_scan1(0)
    if trailing_zeros is None:
        return 0
    else:
        return z.bit_length() - trailing_zeros


def to_shortest_mantissa_exp(r):
    """This destroys info about the precision"""
    m, e = r.as_mantissa_exp()
    trailing_zeros = m.bit_scan1(0)
    if trailing_zeros is None:
        return 0, None
    else:
        return m >> trailing_zeros, e + trailing_zeros


def to_mantissa_exp(r):
    m, e = r.as_mantissa_exp()
    if m == 0:
        return 0, None
    else:
        return m, e


def from_mantissa_exp(m, e):
    if m == 0:
        return mpfr(0, 2)
    mbits = m.bit_length()
    ebits = e.bit_length()
    esig = sigbits(e)
    exp = int(e)
    with exactctx(mbits, min(ebits, mbits, exp), max(ebits, mbits, exp + mbits)):
        rexp = scale = rm = result = None
        try:
            rexp = mpfr(e, max(2, esig))
            scale = mpfr(gmp.exp2(rexp), 2)
            rm = mpfr(m)
            result = gmp.mul(rm, scale)
        except Exception as exc:
            print(exc)
            print(m, e)
            print(mbits, ebits, exp, esig)
            print(result, rm, scale, rexp)
            print(gmp.get_context())
        return result


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
        m = random.randint(0, 1 << 256) << (random.randint(0, 256) if random.randint(0, 2) else 0)
        e = random.randint(-126, 127) if random.randint(0, 2) else random.randint(-(1 << 32), (1 << 32))
        x1 = from_mantissa_exp(m, e)
        x2 = from_mantissa_exp(*to_mantissa_exp(x1))
        if x1 != x2:
            print(m, e, x1, x2)
        if x1.precision != x2.precision:
            print(x1.precision, x2.precision, x1, x2)
    print('...done')


# todo:
# split
# trunc (which is just split)
# interval_contains
# the two arith algos
    
_DEFAULT_PREC = 53
    
class Sink:
    _e : int = None # exponent
    _n : int = None # "sticky bit" or lsb
    _p : int = None # precision: e - n
    _c : int = None # significand
    _negative : bool = None # sign bit
    _inexact : bool = None # approximate bit
    _isinf : bool = None # is the value infinite?
    _isnan : bool = None # is this value NaN?

    
    def _valid(self) -> bool:
        return (
            (self._e >= self._n) and
            (self._p == self._e - self._n) and
            (not (self._isinf and self._isnan))
        )


    def split(self, n):
        """split into exact part with lsb=n and zero with e=lsb=n"""
        
    
    def __init__(self, x, p=None, negative=False, inexact=True) -> None:
        # special case for zeros
        if x is None or x == 0:
            if p is None:
                self._e = self._n = 0
                self._inexact = False
            else:
                self._e = self._n = p
                self._inexact = inexact
            self._p = 0
            self._c = 0
            self._negative = negative
            self._isinf = False
            self._isnan = False
        else:
            prec = _DEFAULT_PREC if p is None else p
            if isinstance(x, str) and x.strip().lower() == 'pi':
                with gmp.context(precision=prec) as gmpctx:
                    r = gmp.const_pi()
                    inexact = True
            else:
                with gmp.context(precision=prec) as gmpctx:
                    r = mpfr(x)
                    # we could infer the needed precision for a literal... but we don't

            m, e = to_mantissa_exp(r)
            # just ignore sign of the mpfr, use given flag
            self._c = abs(int(m))
            self._n = int(e) - 1
            self._p = m.bit_length()
            self._e = self._n + self._p
            self._negative = negative
            self._inexact = inexact
            self._isinf = False
            self._isnan = False

            
    def __repr__(self):
        try:
            mpfr_val = self.as_mpfr()
        except Exception as exc:
            mpfr_val = exc
        try:
            f64_val = self.as_np(np.float64)
        except Exception as exc:
            f64_val = exc
        try:
            f32_val = self.as_np(np.float32)
        except Exception as exc:
            f32_val = exc
        try:
            f16_val = self.as_np(np.float16)
        except Exception as exc:
            f16_val = exc
        return ('Sinking point number:\n  e={}\n  n={}\n  p={}\n  c={}\n  negative={}\n  inexact={}\n  isinf={}\n  isnan={}\n  valid? {}'
                .format(self._e, self._n, self._p, self._c, self._negative, self._inexact, self._isinf, self._isnan, self._valid()) +
                '\n  as mpfr: {}\n  as np.float64: {}\n  as np.float32: {}\n  as np.float16: {}'
                .format(repr(mpfr_val), repr(f64_val), repr(f32_val), repr(f16_val)))


    def as_mpfr(self):
        return from_mantissa_exp(self._c * (-1 if self._negative else 1), self._n + 1)

    
    def as_np(self, ftype=np.float64):
        # fails for 0
        return mkfloat(self._negative, self._e, self._c, ftype=ftype)
            


