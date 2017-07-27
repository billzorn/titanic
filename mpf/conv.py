import sys
import numpy as np
import decimal

from bv import BV
from real import Real
import core

# Binary conversions are relatively simple for numpy's floating point types.
# float16 : w = 5,  p = 11
# float32 : w = 8,  p = 24
# float64 : w = 11, p = 53
# float128: unsupported, not an IEEE 754 128-bit float, possibly 80bit x87?
#           doc says this uses longdouble on the underlying system

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

def np_float_to_packed(f):
    assert isinstance(f, np.float16) or isinstance(f, np.float32) or isinstance(f, np.float64)

    return BV(f.tobytes(), np_byteorder(type(f)))

def np_float_to_implicit(f):
    assert isinstance(f, np.float16) or isinstance(f, np.float32) or isinstance(f, np.float64)

    if isinstance(f, np.float16):
        w = 5
        p = 11
    elif isinstance(f, np.float32):
        w = 8
        p = 24
    else: # isinstance(f, np.float64)
        w = 11
        p = 53

    B = np_float_to_packed(f)
    return core.packed_to_implicit(B, w, p)

def packed_to_np_float(B):
    assert isinstance(B, BV)
    assert B.n == 16 or B.n == 32 or B.n == 64

    if B.n == 16:
        ftype = np.float16
    elif B.n == 32:
        ftype = np.float32
    else: # B.n == 64
        ftype = np.float64

    return np.frombuffer(B.to_bytes(byteorder=np_byteorder(ftype)), dtype=ftype, count=1, offset=0)[0]

def implicit_to_np_float(S, E, T):
    assert isinstance(S, BV)
    assert S.n == 1
    assert isinstance(E, BV)
    assert isinstance(T, BV)
    assert (E.n == 5 and T.n == 10) or (E.n == 8 and T.n == 23) or (E.n == 11 and T.n == 52)

    return packed_to_np_float(core.implicit_to_packed(S, E, T))

# Python's built-in float is a double, so we can treat is as a numpy float64.

def float_to_packed(f):
    assert isinstance(f, float)

    return np_float_to_packed(np.float64(f))

def float_to_implicit(f):
    assert isinstance(f, float)

    return np_float_to_implicit(np.float64(f))

def packed_to_float(B):
    assert isinstance(B, BV)
    assert B.n == 64

    return float(packed_to_np_float(B))

def implicit_to_float(S, E, T):
    assert isinstance(S, BV)
    assert S.n == 1
    assert isinstance(E, BV)
    assert isinstance(T, BV)
    assert E.n == 11 and T.n == 52

    return float(implicit_to_np_float(S, E, T))

# Conversions to human-readable formats.

def str_to_implicit(s, w, p, rm = core.RNE):
    assert isinstance(s, str)
    assert isinstance(w, int)
    assert w >= 2
    assert isinstance(p, int)
    assert p >= 2
    assert rm == core.RTN or rm == core.RTP or rm == core.RTZ or rm == core.RNE or rm == core.RNA

    sl = s.strip().lower()
    if sl.startswith('-'):
        Ss = BV(1, 1)
    else:
        Ss = BV(0, 1)

    r = Real(s)
    S, E, T = core.real_to_implicit(r, w, p, rm)

    # In the case of nan or 0, override the provided sign with what we
    # saw in the string.
    if r.isnan or r == Real(0):
        return Ss, E, T
    else:
        return S, E, T

def real_to_dec(r, prec, S = BV(0, 1)):
    assert isinstance(r, Real)
    assert isinstance(prec, int)
    assert prec >= 1
    
    if r.isnan:
        if S.uint == 0:
            return decimal.Decimal('nan')
        else:
            return decimal.Decimal('-nan')
    elif r.isinf:
        if S.uint == 0:
            return decimal.Decimal('inf')
        else:
            return decimal.Decimal('-inf')
    elif r == Real(0):
        if S.uint == 0:
            return decimal.Decimal('0')
        else:
            return decimal.Decimal('-0')
    else:
        decimal.getcontext().prec = prec
        d = decimal.Decimal(r.numerator) / decimal.Decimal(r.denominator)
        return d


# This uses larger decimals than strictly necessary, but that's
# ok, just inefficient.
def implicit_to_dec(S, E, T):
    assert isinstance(S, BV)
    assert S.n == 1
    assert isinstance(E, BV)
    assert E.n >= 2
    assert isinstance(T, BV)
    assert T.n >= 1
    
    prec = max(28, 2 ** E.n, (T.n + 1) * 2)
    
    r = core.implicit_to_real(S, E, T)
    return real_to_dec(r, prec, S)