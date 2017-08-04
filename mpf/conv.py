import sys
import numpy as np
import sympy

from bv import BV
import real
FReal = real.FReal
import core
import re

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

# Pre-rounding conversion to bounded reals.

def ordinal_to_bounded_real(i, w, p):
    assert isinstance(i, int)
    assert isinstance(w, int)
    assert w >= 2
    assert isinstance(p, int)
    assert p >= 2
    umax = ((2 ** w) - 1) * (2 ** (p - 1))
    assert -umax <= i and i <= umax

    below = i
    above = i
    S, E, T = core.ordinal_to_implicit(x, w, p)
    R = core.implicit_to_real(S, E, T)
    return R, below, above

def bv_to_bounded_real(B, w, p):
    assert isinstance(B, BV)
    assert isinstance(w, int)
    assert w >= 2
    assert isinstance(p, int)
    assert p >= 2
    assert B.n == w + p

    try:
        below = core.packed_to_ordinal(x, w, p)
    except core.OrdinalError:
        below = None
    above = below
    S, E, T = core.packed_to_implicit(x, w, p)
    R = core.implicit_to_real(S, E, T)
    return R, below, above

def real_to_bounded_real(R, w, p):
    assert isinstance(R, FReal)
    assert isinstance(w, int)
    assert w >= 2
    assert isinstance(p, int)
    assert p >= 2

    if R.isnan:
        return R, None, None
    else:
        below, above = core.binsearch_nearest_ordinals(R, w, p)
        return R, below, above

# Conversions to and from human-readable formats.

# custom string parser for ordinals and bitvectors
def str_to_ord_bv_real(x):
    assert isinstance(x, str)

    s = x.strip().lower()
    # ordinal
    if s.startswith('0i'):
        s = s[2:]
        return int(s)
    # hex bitvector
    elif s.startswith('0x'):
        s = s[2:]
        b = int(s, 16)
        n = len(s) * 4
        return BV(b, n)
    # binary bitvector
    elif s.startswith('0b'):
        s = s[2:]
        b = int(s, 2)
        n = len(s)
        return BV(b, n)
    # see if the FReal constructor can figure it out
    else:
        return FReal(x)

# convenient; mostly equivalent to float(s) but it returns an implicit triple
def str_to_implicit(s, w, p, rm = core.RNE):
    assert isinstance(s, str)
    assert isinstance(w, int)
    assert w >= 2
    assert isinstance(p, int)
    assert p >= 2
    assert rm == core.RTN or rm == core.RTP or rm == core.RTZ or rm == core.RNE or rm == core.RNA

    r = FReal(s)
    return core.real_to_implicit(r, w, p, rm)

# Returns integers c an e such that c * (10**e) == R.
# Note that the sign of zero is destroyed.
# If no such c and e exist, then returns (None, None)
def real_to_pow10(R):
    assert isinstance(R, FReal)
    if not R.isrational:
        return None, None

    p = R.rational_numerator
    q = R.rational_denominator

    q_factors = sympy.factorint(q, limit=5)
    if not all(k == 2 or k == 5 for k in q_factors):
        return None, None

    # factor
    q_2 = q_factors.get(2, 0)
    q_5 = q_factors.get(5, 0)
    p_factors = sympy.factorint(p, limit=5)
    p_2 = p_factors.get(2, 0)
    p_5 = p_factors.get(5, 0)

    # reduce
    reduce_2 = min(q_2, p_2)
    reduce_5 = min(q_5, p_5)
    if reduce_2 > 0 or reduce_5 > 0:
        reduce_10 = (2 ** reduce_2) * (5 ** reduce_5)
        p_new = p // reduce_10
        q_new = q // reduce_10
        assert p_new * reduce_10 == p
        assert q_new * reduce_10 == q
        p = p_new
        q = q_new
        q_2 = q_2 - reduce_2
        q_5 = q_5 - reduce_5
        p_2 = p_2 - reduce_2
        p_5 = p_5 - reduce_5

    # scale so the denominator is divisible by 10
    skew = q_2 - q_5
    if skew <= 0:
        scale = 2 ** (-skew)
        p_2 = p_2 - skew
    else:
        scale = 5 ** skew
        p_5 = p_5 + skew
        # q_2 and q_5 aren'd used again

    p_scaled = p * scale
    q_scaled = q * scale

    # negative powers of 10
    pow_10 = sympy.log(q_scaled, 10)
    assert pow_10.is_integer

    if pow_10 > 0:
        c = int(p_scaled)
        e = -int(pow_10)
    else:
        # positive powers of 10
        e = int(min(p_2, p_5))
        assert e >= 0
        c = int(p_scaled // (10 ** e))

    assert R == FReal(c) * (FReal(10) ** e)
    return c, e

def pow10_to_f_str(c, e):
    assert isinstance(c, int)
    assert isinstance(e, int)

    if c < 0:
        sign_str = '-'
        c = -c
    else:
        sign_str = ''

    if e < 0:
        s = str(c)
        left = s[:e]
        if len(left) == 0:
            left = '0'
        right = s[e:]
        if len(right) < (-e):
            right = ((-len(right) - e) * '0') + right
        return sign_str + left + '.' + right
    elif e == 0:
        return sign_str + str(c)
    else:
        return sign_str + str(c) + ('0' * e)

def pow10_to_e_str(c, e):
    assert isinstance(c, int)
    assert isinstance(e, int)

    if c < 0:
        sign_str = '-'
        c = -c
    else:
        sign_str = ''

    s = str(c)
    e2 = len(s) - 1
    if e2 <= 0:
        if e < 0:
            esign_str = ''
        else:
            esign_str = '+'
        return sign_str + s + 'e' + esign_str + str(e)
    else:
        e3 = e + e2
        if e3 < 0:
            esign_str = ''
        else:
            esign_str = '+'
        return sign_str + s[:1] + '.' + s[1:] + 'e' + esign_str + str(e3)

default_prec = 12
approx_str = u'\u2248'
dec_re = re.compile(r'[-+]?([0-9]+)\.?([0-9]*)([eE][-+]?[0-9]+)?')

# "Character precision" of a decimal string in standard of scientific format.
# There MUST be something (such as 0) before the decimal point. Returns 0 if the
# format is not recognized.
def sprec_of(s):
    assert isinstance(s, str)
    m = dec_re.fullmatch(s)
    if m:
        return(len(m.group(1) + m.group(2)))
    else:
        return 0

# "Decimal precision" of a number c * (10**e). If e is not given,
# this counts the number of digits in c. If e is given, this gives
# the "character precision" of the standard (non-scientific) representation.
# c must not be divisible by 10. 0 always has precision 1.
def prec_of(c, e = None):
    assert isinstance(c, int)
    assert c % 10 != 0
    assert isinstance(e, int) or e is None

    if c == 0:
        return 1
    c = abs(c)
    prec = sympy.floor(sympy.log(c, 10)) + 1

    if e is None:
        return prec
    else:
        if e > 0:
            return prec + e
        else:
            return max(prec, 1 - e)

# If exact is true, then the precise value should be recoverable from the
# output. If not, we will try to respect the character limit n, and prefix
# the string with a unicode u"\u2248"
def real_to_string(R, prec = default_prec, exact = True, exp = None, show_payload = False):
    assert isinstance(R, FReal)
    assert isinstance(prec, int)
    assert prec >= 0
    assert exact is True or exact is False
    assert exp is True or exp is False or exp is None
    assert show_payload is True or show_payload is False

    if R.isnan:
        if show_payload:
            return real.preferred_nan_str + str(R.nan_payload)
        else:
            return real.preferred_nan_str
    elif R.isinf or R.iszero:
        return str(R)
    else:
        c, e = real_to_pow10(R)

        # no exact decimal representation
        if c is None or e is None:
            if exact:
                return str(R.symbolic_value)
            else:
                return approx_str + str(R.evaluate(prec, abort_incomparables=False))
        # there is an exact decimal representation
        else:
            # Exact representation: choose based on exp first, or
            # heuristically better if exp is None.
            if exact:
                if exp is True:
                    return pow10_to_e_str(c, e)
                elif exp is False:
                    return pow10_to_f_str(c, e)
                else: # exp is None
                    sprec = prec_of(c, e)
                    eprec = prec_of(c)
                    if sprec <= prec or sprec <= eprec + 3:
                        return pow10_to_f_str(c, e)
                    else:
                        return pow10_to_e_str(c, e)
            else:
                # We're giving an approximate representation;
                # first see if there is a short exact one, otherwise
                # truncate with evalf.
                sprec = prec_of(c, e)
                eprec = prec_of(c)
                if sprec <= prec:
                    return pow10_to_f_str(c, e)
                elif eprec <= prec:
                    return pow10_to_e_str(c, e)
                else:
                    return approx_str + str(R.evaluate(prec, abort_incomparables=False))

# Produce a unicode rendering with sympy.pretty. This is probably
# not able to be parsed back in.
def real_to_pretty_string(R):
    assert isinstance(R, FReal)

    if R.isnan:
        return 'NaN'
    elif R.isinf:
        return sympy.pretty(sympy.oo * R.sign)
    elif R.iszero:
        return str(R)
    else:
        c, e = real_to_pow10(R)
        if c is None or e is None:
            return sympy.pretty(R.symbolic_value)
        else:
            # abuse precision as a heuristic for determining the "most readable" form
            sprec = prec_of(c, e)
            eprec = prec_of(c)
            if sprec <= default_prec:
                return pow10_to_f_str(c, e)
            elif eprec <= default_prec:
                return pow10_to_e_str(c, e)
            else:
                return sympy.pretty(R.symbolic_value)
