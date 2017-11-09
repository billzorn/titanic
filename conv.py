import sys
import operator
import re
import numpy as np
import sympy

import reparse
Result = reparse.Result
from bv import BV
import real
FReal = real.FReal
import core

# work around sympy floor bug:
def floor_log10(r, maxn = real.default_maxn):
    assert isinstance(r, int) or r.is_real
    assert isinstance(maxn, int)
    assert maxn > 0

    if isinstance(r, int):
        if r < 0:
            sign = -1
        elif r == 0:
            sign = 0
        else:
            sign = 1
    else:
        sign = r.evalf(2, maxn=maxn)

    if not (isinstance(sign, int) or sign.is_comparable):
        raise ValueError('floor_log10: unable to determine sign of {}, maxn={}'
                         .format(repr(r), repr(maxn)))
    elif sign == 0:
        raise ValueError('floor_log10: log of zero {}, maxn={}'
                         .format(repr(r), repr(maxn)))
    elif sign < 0:
        r = -r

    log10f = sympy.log(r, 10).evalf(20, maxn=maxn)

    if log10f > sympy.Float('1e19', 20):
        raise ValueError('floor_log10: log overflow for {}, maxn={}'
                         .format(repr(r), repr(maxn)))
    else:
        return int(sympy.floor(log10f))

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

# Conversions to and from human-readable formats.

# Try to interpret a string as either a number literal, or a discrete fp representation.
# This will work for anything recognized by the reparse regex parser (i.e. not sympy expressions).
# If we see a discrete type like an ordinal, triple, or bitvector, we will return an implicit triple
# S, E, T. If we see a number literal, we will return an FReal. Note that w and p may change, if the literal
# could be interpreted with different w and p from the ones provided. We assume a tuple is a written implicit
# triple, unless w and p are exactly right for it to be an explicit triple.
def str_to_real_or_implicit(s, w, p, limit_exp=None):
    assert isinstance(s, str)
    assert isinstance(w, int)
    assert w >= 2
    assert isinstance(p, int)
    assert p >= 2
    assert isinstance(limit_exp, int) or limit_exp is None
    assert limit_exp is None or limit_exp >= 0

    res, xs = reparse.reparse(s)

    # continuous - return real

    if res is Result.NAN:
        (sign, p,) = xs
        if p is None:
            return False, FReal(None, negative=sign<0)
        elif p == 0:
            return None, 'NaN payload must not be 0 (that would be an infinity!).'
        else:
            return False, FReal(None, negative=sign<0, payload=p)

    elif res is Result.INF:
        (sign,) = xs
        return False, FReal(None, negative=sign<0, infinite=True)

    elif res is Result.FPC:
        # give up and parse again...
        return False, FReal(s)

    elif res is Result.NUM:
        (sign, top, bot, base, exp,) = xs
        if limit_exp and exp is not None:
            if base == 10:
                effective_exp = exp
            else:
                effective_exp = (FReal(10) / FReal(base)) * FReal(abs(exp))
            if effective_exp > limit_exp:
                return None, 'Effective exponent 10**({}) must be less than {}.'.format(str(effective_exp), str(limit_exp))
        frac = real.Rational(top, bot)
        if exp is not None:
            frac = frac * (real.Rational(base) ** exp)
        return False, FReal(frac, negative=sign<0)

    # discrete - return implicit triple

    elif res is Result.ORD:
        (i,) = xs
        return True, core.ordinal_to_implicit(i, w, p)

    elif res is Result.BV:
        (v, size,) = xs
        if size == w + p:
            return True, core.packed_to_implicit(BV(v, size), w, p)
        else:
            w_prime, p_prime = ieee_split_w_p(size)
            if w_prime is not None and p_prime is not None:
                return True, core.packed_to_implicit(BV(v, size), w_prime, p_prime)
            else:
                return None, ('Unable to interpret bitvector {} as binary floating point representation with w={:d}, p={:d}.'
                              .format(repr(s), w, p))

    elif res is Result.ITUP:
        (S_tup, E_tup, T_tup,) = xs
        S = BV(*S_tup)
        E = BV(*E_tup)
        T = BV(*T_tup)
        if E.n < 2:
            return None, 'Exponent must have at least 2 bits'
        else:
            return True, (S, E, T,)

    elif res is Result.ETUP:
        (S_tup, E_tup, ibit_tup, T_tup,) = xs
        S = BV(*S_tup)
        E = BV(*E_tup)
        ibit = BV(*ibit_tup)
        T = BV(*T_tup)
        C = core.concat(ibit, T)
        if E.n < 2:
            return None, 'Exponent must have at least 2 bits'
        else:
            return True, core.explicit_to_implicit(S, E, C)

    else:
        return None, None

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

# convert a decimal representation c * (10**e) == R
# back to the real number R.
# there are a few special cases:
#   if c is None, then this pair is no good (return None)
#   if e is None, then this number is not real:
#     if c is 0, then this is some NaN (return FReal(None), payload of nan is not stored)
#     if c is not 0, then this number is inf (return c * FReal(infinite=True))
def pow10_to_real(c, e):
    assert isinstance(c, int) or c is None
    assert isinstance(e, int) or e is None

    if c is None:
        return None
    elif e is None:
        if c == 0:
            return FReal(None)
        else:
            return FReal(c) * FReal(infinite=True)
    else:
        return FReal(c) * (FReal(10) ** e)

# convert a decimal representation with a real-valued remainder c * (10**e) + rem == R
# back to the real number R.
# if c, e is no good, return the remainder directly.
# if the remainder is None, then ignore it.
# if c * (10**e) is 0, then be clever and return the remainder directly to retain the sign
# of zero if we can.
def pow10_rem_to_real(c, e, rem):
    assert isinstance(c, int) or c is None
    assert isinstance(e, int) or e is None
    assert isinstance(rem, FReal) or rem is None

    if c is None:
        return rem
    else:
        R = pow10_to_real(c, e)
        if rem is None:
            return R
        elif R is None or R.iszero:
            return rem
        else:
            return R + rem

# Returns integers c an e such that c * (10**e) == R.
# Note that the sign of zero is destroyed.
# If no such c and e exist, then returns (None, None)
def real_to_pow10(R):
    assert isinstance(R, FReal)

    if R.isnan:
        return 0, None
    elif R.isinf:
        return R.sign, None
    elif not R.isrational:
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

    return c, e

# Returns an approximate c and e, and remainder rem, such that c*(10**e) + rem == R.
# c will have exactly n digits.
# If c and e are exact, then rem will be FReal(0). If R was -0, rem will be FReal(0, negative=True).
# NaN has a decimal representation but no meaningful remainder, so remainder will be None.
def real_to_pow10_rem(R, n):
    assert isinstance(R, FReal)
    assert isinstance(n, int)
    assert n >= 1

    if R.isnan:
        return 0, None, None
    elif R.isinf:
        return R.sign, None, FReal(0)
    elif R.iszero:
        return 0, 0, R
    else:
        f = R.numeric_value(n)
        e_f = floor_log10(f)

        # get f to be an integer, adjust e
        e_scale = n - (e_f + 1)
        scale = 10 ** abs(e_scale)
        if e_scale > 0:
            c = int(f * scale)
        else:
            c = int(f / scale)
        e = e_f - n + 1

        # it's possible we didn't round correctly, so do that
        # (though this is very, very slow)
        c_lo = c - 1
        R_lo_approx = FReal(c_lo) * (FReal(10) ** e)
        R_approx = FReal(c) * (FReal(10) ** e)
        c_hi = c + 1
        R_hi_approx = FReal(c_hi) * (FReal(10) ** e)

        remlo = R - R_lo_approx
        remmid = R - R_approx
        remhi = R - R_hi_approx

        rem_abs, c_e_rem = min(
            (abs(remlo),  (c_lo, e, remlo,), ),
            (abs(remmid), (c,    e, remmid,),),
            (abs(remhi),  (c_hi, e, remhi,), ),
            key=operator.itemgetter(0))

        return c_e_rem

def pow10_inc(c, e, negative = False):
    assert isinstance(c, int)
    assert isinstance(e, int) or e is None
    assert negative is True or negative is False

    # no next/previous number for infinity and NaN
    if e is None:
        return c, e
    elif negative:
        if c == 1:
            return 9, e-1
        else:
            return c-1, e
    # might result in a number with more digits, expensive to check
    else:
        return c+1, e

def pow10_to_f_str(c, e):
    assert isinstance(c, int) or c is None
    assert isinstance(e, int) or e is None

    if c is None:
        return None

    if c < 0:
        sign_str = '-'
        c = -c
    else:
        sign_str = ''

    if e is None:
        if c == 0:
            return reparse.preferred_nan_str
        else:
            return sign_str + reparse.preferred_inf_str

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
    assert isinstance(c, int) or c is None
    assert isinstance(e, int) or e is None

    if c is None:
        return None

    if c < 0:
        sign_str = '-'
        c = -c
    else:
        sign_str = ''

    if e is None:
        if c == 0:
            return reparse.preferred_nan_str
        else:
            return sign_str + reparse.preferred_inf_str

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
bonus_prec = 4
approx_str = u'\u2248'
dec_re = re.compile(r'[-+]?([0-9]+)\.?([0-9]*)([eE][-+]?[0-9]+)?')

# Number of decimal digits in an integer.
def ndig(c):
    assert isinstance(c, int) or c is None

    if c is None:
        return 0
    elif c == 0:
        return 1
    else:
        return floor_log10(c) + 1

# "Character precision" of a decimal string in standard or scientific format.
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
    assert isinstance(c, int) or c is None
    assert isinstance(e, int) or e is None

    prec = ndig(c)
    if e is None or c is None:
        return prec
    else:
        if e > 0:
            return prec + e
        else:
            return max(prec, 1 - e)

# slow, destroys information, useful for consistent printing
def simplify_exponent(c, e):
    assert isinstance(c, int) or c is None
    assert isinstance(e, int) or e is None

    if c is None:
        return None, None
    elif e is None:
        if c > 0:
            return 1, None
        elif c == 0:
            return 0, None
        else: # c < 0
            return -1, None
    elif c == 0:
        return 0, 0
    else:
        while c % 10 == 0:
            c = c // 10
            e = e + 1
        return c, e

# abuse precision as a heuristic for determining the "most readable" form
def pow10_to_str(c, e, simplify = True):
    assert isinstance(c, int) or c is None
    assert isinstance(e, int) or e is None
    assert simplify is True or simplift is False

    if simplify:
        c, e = simplify_exponent(c, e)
    sprec = prec_of(c, e)
    eprec = prec_of(c)
    if sprec <= eprec + bonus_prec:
        return pow10_to_f_str(c, e)
    else:
        return pow10_to_e_str(c, e)

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
            return reparse.preferred_nan_str + str(R.nan_payload)
        else:
            return reparse.preferred_nan_str
    elif R.isinf or R.iszero:
        return str(R)
    else:
        c, e = real_to_pow10(R)

        # no exact decimal representation
        if c is None or e is None:
            if exact:
                return str(R.symbolic_value)
            else:
                return approx_str + str(R.numeric_value(prec, abort_incomparables=False))
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
                    return pow10_to_str(c, e)
            else:
                # We're giving an approximate representation;
                # first see if there is a short exact one, otherwise
                # truncate with evalf.
                sprec = prec_of(c, e)
                eprec = prec_of(c)
                if sprec <= prec + bonus_prec:
                    return pow10_to_f_str(c, e)
                elif eprec <= prec:
                    return pow10_to_e_str(c, e)
                else:
                    # TODO: should this use quantize_dec?
                    return approx_str + str(R.numeric_value(prec, abort_incomparables=False))

# Produce a unicode rendering with sympy.pretty. This is probably
# not able to be parsed back in.
def real_to_pretty_string(R, num_columns = 100):
    assert isinstance(R, FReal)

    if R.isnan:
        return 'NaN'
    elif R.isinf:
        return sympy.pretty(sympy.oo * R.sign, num_columns=num_columns)
    elif R.iszero:
        return str(R)
    else:
        c, e = real_to_pow10(R)
        if c is None or e is None:
            return sympy.pretty(R.symbolic_value, num_columns=num_columns)
        else:
            sprec = prec_of(c, e)
            eprec = prec_of(c)
            if sprec <= default_prec:
                return pow10_to_f_str(c, e)
            elif eprec <= default_prec:
                return pow10_to_e_str(c, e)
            else:
                return sympy.pretty(R.symbolic_value, num_columns=num_columns)

# The "rounding envelope" of some F = (S, E, T) under rm.
# return lower, lower_inclusive, upper, upper_inclusive,
# such that all real values r in the range:
# lower < r < upper, or r = lower if lower_inclusive, or r = upper if upper_inclusive,
# will round to F under rm.
def implicit_to_rounding_envelope(S, E, T, rm):
    assert isinstance(S, BV)
    assert S.n == 1
    assert isinstance(E, BV)
    assert E.n >= 2
    assert isinstance(T, BV)
    assert rm == core.RTN or rm == core.RTP or rm == core.RTZ or rm == core.RNE or rm == core.RNA

    R = core.implicit_to_real(S, E, T)
    assert not R.isnan

    w = E.n
    p = T.n + 1
    umax = ((2 ** w) - 1) * (2 ** (p - 1))

    i = core.implicit_to_ordinal(S, E, T)
    i_prev = max(i-1, -umax)
    R_prev = core.implicit_to_real(*core.ordinal_to_implicit(i_prev, w, p))
    i_next = min(i+1, umax)
    R_next = core.implicit_to_real(*core.ordinal_to_implicit(i_next, w, p))

    # -0 compliant nextafter behavior
    if R_next.iszero:
        R_next = -R_next

    emax = (2 ** (w - 1)) - 1
    fmax = (FReal(2) ** emax) * (FReal(2) - ((FReal(2) ** (1 - p)) / FReal(2)))

    if i == -umax:
        lower = FReal(negative=True, infinite=True)
    elif i == -(umax - 1):
        lower = -fmax
    elif i == umax:
        lower = fmax
    else:
        lower = R_prev + ((R - R_prev) / 2)

    if i == -umax:
        upper = -fmax
    elif i == umax:
        upper = FReal(negative=False, infinite=True)
    elif i == umax - 1:
        upper = fmax
    else:
        upper = R + ((R_next - R) / 2)

    # we need to be careful with the edges to avoid returning the empty envelope (x, x)
    if rm == core.RTN:
        include_boundaries = R == R_next
        return R, True, R_next, include_boundaries
    elif rm == core.RTP:
        include_boundaries = R_prev == R
        return R_prev, include_boundaries, R, True
    elif rm == core.RTZ:
        if R < 0:
            include_boundaries = R_prev == R
            return R_prev, include_boundaries, R, True
        elif R == 0:
            include_boundaries = R_prev == R_next
            return R_prev, include_boundaries, R_next, include_boundaries
        else:
            include_boundaries = R == R_next
            return R, True, R_next, include_boundaries
    elif rm == core.RNE:
        # case inf: round towards (good)
        # case largest finite: round away (good)
        include_boundaries = T[0] == 0
        return lower, include_boundaries, upper, include_boundaries
    else: # rm == core.RNA:
        # Here, we don't have anything "past" infinity that we would round a true
        # infinity away towards, so remember to make those boundaries inclusive.
        if R < 0:
            return lower, lower.isinf, upper, True
        elif R == 0:
            include_boundaries = lower == upper
            return lower, include_boundaries, upper, include_boundaries
        else:
            return lower, True, upper, upper.isinf

# Quantize c to exactly n decimal places:
# return c_lo, c_mid, c_hi, e_prime, s.t.
#   c_lo*(10**e_prime) <= c*(10**e) <= c_hi*(10**e_prime) and
#   c_lo*(10**e_prime) <= c_mid*(10**e_prime) <= c_hi*(10**e_prime) and
#   c_hi - c_lo <= 1 and
#   and c_mid is correctly rounded, or None if exactly between.
# If either c_lo or c_hi is exact, then they must be the same.
# intrem is a sub-integral remainer used to break ties: we're really comparing
# to some real number c*(10**e) + intrem, so if the integral remainder c_rem is 0,
# then we also check intrem.
def quantize_dec(c, e, n, intrem = 0):
    assert isinstance(c, int)
    assert isinstance(e, int) or e is None
    assert isinstance(n, int)
    assert n >= 1
    assert isinstance(intrem, int)
    assert -1 <= intrem and intrem <= 1

    if e is None:
       if c > 0:
           return 1, 1, 1, None
       elif c == 0:
           return 0, 0, 0, None
       else: # c < 0
           return -1, -1, -1, None

    prec = ndig(c)
    if prec <= n:
        scale_e = n - prec
        scale = 10 ** scale_e
        c_scaled = c * scale
        return c_scaled, c_scaled, c_scaled, e - scale_e
    else:
        scale_e = prec - n
        scale = 10 ** scale_e

        e_prime = e + scale_e

        c_floor = c // scale
        c_rem = c % scale

        if c_rem == 0 and intrem == 0:
            return c_floor, c_floor, c_floor, e_prime

        round_comp = c_rem - (scale // 2)

        # round down
        if round_comp < 0:
            return c_floor, c_floor, c_floor + 1, e_prime
        # exactly between
        elif round_comp == 0:
            if intrem < 0:
                return c_floor, c_floor, c_floor + 1, e_prime
            elif intrem == 0:
                return c_floor, None, c_floor + 1, e_prime
            else:
                return c_floor, c_floor + 1, c_floor + 1, e_prime
        # round up
        else:
            return c_floor, c_floor + 1, c_floor + 1, e_prime

# Find the shortest prefix of (c*(10**e)) + (epsilon<<1)*intrem) that fits in
# the envelope defined by lower, lower_inclusive, upper, upper_inclusive.
# If round_correctly is True, then ensure this prefix is correctly rounded,
# i.e. 1.27 can only be rounded to 1.3, never 1.2.
# return a bunch of numbers describing the precision and the decimals of that
# precision that fall inside the envelope:
#   prec, lowest_ce, midlo_ce, midhi_ce, highest_ce
# As a hack to allow infinite c, e values, c=+-1, e=None indicates an infinity of the corresponding sign.
def binsearch_shortest_dec(c, e, intrem,
                           lower, lower_inclusive, upper, upper_inclusive,
                           round_correctly = False):
    assert isinstance(c, int)
    assert isinstance(e, int) or e is None
    assert not lower.isnan
    assert lower_inclusive is True or lower_inclusive is False
    assert not upper.isnan
    assert upper_inclusive is True or upper_inclusive is False
    R = pow10_to_real(c, e)
    assert (lower < R and R < upper) or (lower_inclusive and lower == R) or (upper_inclusive and R == upper)
    assert round_correctly is True or round_correctly is False

    below = 1
    above = ndig(c)

    # already have few enough digits
    if above <= below:
        c_midlo = c
        c_midhi = c
        e_mid = e

    # binary search!

    # Question: is this procedure actually stable?
    # or can we find some n, such that quantize(c, e, n) is not in the envelope,
    # and yet there exists n' < n such that quantize(c, e, n') is.

    # I think it is stable, if you allow incorrect rounding here.

    # Say we quantize to n, and neither c_lo or c_hi puts us in the envelope,
    # but there is some n' < n we could round to that puts us back in the envelope.
    # Quantizing to a smaller n' will produce c_lo' <= c_lo and c_hi' >= c_hi. So
    # we have a contradiction: c_lo or c_hi had to be in the envelope as well. They
    # are the two closest numbers to the original number at precision n *or any lower
    # precision*. Note that this proof fails if we are forced to quantize to c_mid, as
    # our target point could be off center in the envelope, and the "wrong" rounding
    # direction could be the only one that stays in.
    found_c = False
    while below < above:
        between = below + ((above - below) // 2)
        c_lo, c_mid, c_hi, e_prime = quantize_dec(c, e, between, intrem=intrem)

        lo_ok = False
        hi_ok = False

        # This procedure can round INCORRECTLY, if that's what it takes to stay in the envelope.
        # This means we'll always find the shortest string (proof?????) but it might not
        # be quite the one you thought you were going to get.

        R_lo = pow10_to_real(c_lo, e_prime)
        R_hi = pow10_to_real(c_hi, e_prime)

        if ((lower < R_lo and R_lo < upper) or
            (lower_inclusive and lower == R_lo) or
            (upper_inclusive and R_lo == upper)):
            lo_ok = True

            c_midlo = c_lo
            c_midhi = c_lo
            e_mid = e_prime

        if ((lower < R_hi and R_hi < upper) or
            (lower_inclusive and lower == R_hi) or
            (upper_inclusive and R_hi == upper)):
            hi_ok = True

            if not lo_ok:
                c_midlo = c_hi
                c_midhi = c_hi
                e_mid = e_prime
            else:
                if c_mid is None:
                    c_midhi = c_hi
                else:
                    c_midlo = c_mid
                    c_midhi = c_mid

        if lo_ok or hi_ok:
            found_c = True
            above = between
        else:
            below = between + 1

    # Search might terminate without moving above down, or ever checking it
    # to produce c_midlo, etc.
    # Handle that case here.
    if not found_c:
        c_midlo = c
        c_midhi = c
        e_mid = e

    assert above <= below
    prec = above

    # We may have rounded incorrectly. Linearly scan up to find the shortest (maybe longer)
    # prefix that is correctly rounded and still fits in the envelope. Usually this shouldn't
    # take too long, but I have absolutely no proof of that.
    if round_correctly:
        scan_shortest_correct = True

        while scan_shortest_correct:
            c_lo, c_mid, c_hi, e_prime = quantize_dec(c, e, prec, intrem=intrem)

            mid_ok = False
            lo_ok = False
            hi_ok = False

            # To enforce correct rounding, we have to use c_mid when it's not None.
            if c_mid is not None:
                R_mid = pow10_to_real(c_mid, e_prime)

                if ((lower < R_mid and R_mid < upper) or
                    (lower_inclusive and lower == R_mid) or
                    (upper_inclusive and R_mid == upper)):
                    mid_ok = True

                    c_midlo = c_mid
                    c_midhi = c_mid
                    e_mid = e_prime

            else:
                R_lo = pow10_to_real(c_lo, e_prime)
                R_hi = pow10_to_real(c_hi, e_prime)

                if ((lower < R_lo and R_lo < upper) or
                    (lower_inclusive and lower == R_lo) or
                    (upper_inclusive and R_lo == upper)):
                    lo_ok = True

                    c_midlo = c_lo
                    c_midhi = c_lo
                    e_mid = e_prime

                if ((lower < R_hi and R_hi < upper) or
                    (lower_inclusive and lower == R_hi) or
                    (upper_inclusive and R_hi == upper)):
                    hi_ok = True

                    if not lo_ok:
                        c_midlo = c_hi
                        c_midhi = c_hi
                        e_mid = e_prime
                    else:
                        if c_mid is None:
                            c_midhi = c_hi
                        else:
                            c_midlo = c_mid
                            c_midhi = c_mid

            if mid_ok or lo_ok or hi_ok:
                scan_shortest_correct = False
            else:
                prec = prec + 1

    # look for lowest and highest
    c_lower, e_lower = real_to_pow10(lower)
    assert c_lower is not None
    if e_lower is None:
        # should never be NaN
        assert c_lower != 0
        # if this is a closed interval, return the infinity
        if lower_inclusive:
            c_lowest, e_lowest = c_lower, e_lower
        # if this is an open interval, then there is no next number closest to inf
        else:
            c_lowest, e_lowest = None, None
    else:
        c_lo, c_mid, c_hi, e_prime = quantize_dec(c_lower, e_lower, prec)
        R_hi = pow10_to_real(c_hi, e_prime)
        if lower < R_hi or (lower_inclusive and lower == R_hi):
            c_lowest, e_lowest = c_hi, e_prime
        else:
            c_lowest, e_lowest = pow10_inc(c_hi, e_prime, negative=False)
            # is this needed?
            R_hi = pow10_to_real(c_lowest, e_lowest)
            assert lower < R_hi or (lower_inclusive and lower == R_hi)

    c_upper, e_upper = real_to_pow10(upper)
    if e_upper is None:
        # should never be NaN
        assert c_upper != 0
        # if this is a closed interval, return the infinity
        if upper_inclusive:
            c_highest, e_highest = c_upper, e_upper
        # if this is an open interval, then there is no next number closest to inf
        else:
            c_highest, e_highest = None, None
    else:
        c_lo, c_mid, c_hi, e_prime = quantize_dec(c_upper, e_upper, prec)
        R_lo = pow10_to_real(c_lo, e_prime)
        if R_lo < upper or (upper_inclusive and R_lo == upper):
            c_highest, e_highest = c_lo, e_prime
        else:
            c_highest, e_highest = pow10_inc(c_lo, e_prime, negative=True)
            # is this needed?
            R_lo = pow10_to_real(c_highest, e_highest)
            assert R_lo < upper or (upper_inclusive and R_lo == upper)

    return prec, (c_lowest, e_lowest,), (c_midlo, e_mid,), (c_midhi, e_mid,), (c_highest, e_highest,)

# Find the shortest decimal numerator that can be used to recover R = (S, E, T) under rm.
# There might be multiple such numerators; if so report all of them.
# Specifically, we return (prec, lowest, midlo, midhi, highest, e) where:
#  prec    : int is the minimal precision needed to recover E and T under rm.
#  lowest  : int is the smallest decimal numerator of that precision that ever rounds to F.
#  midlo   : int is the smaller decimal numerator that is as close to the real value of F as possible.
#  midhi   : int is as midlo, but the larger one. The same as midlo if one numerator is closest.
#  highest : int is the largest decimal numerator of that precision that ever rounds to F.
#  e       : int is the exponent, such that F = round({lowest,midlo,midhi,highest} * (10**e))
# If F is nan, then return a precision of 0 and a bunch of Nones.
# if F is zero or inf, then do something special.
def shortest_dec(R, S, E, T, rm, round_correctly = False):
    assert isinstance(R, FReal)
    assert isinstance(S, BV)
    assert S.n == 1
    assert isinstance(E, BV)
    assert E.n >= 2
    assert isinstance(T, BV)
    assert rm == core.RTN or rm == core.RTP or rm == core.RTZ or rm == core.RNE or rm == core.RNA
    assert round_correctly is True or round_correctly is False

    if R.isnan:
        assert core.implicit_to_real(S, E, T).isnan
        return 0, None, None, None, None, None
    else:
        lower, lower_inclusive, upper, upper_inclusive = implicit_to_rounding_envelope(S, E, T, rm)
        assert (lower < R and R < upper) or (lower_inclusive and lower == R) or (upper_inclusive and R == upper)

        if R.isrational:
            c, e = real_to_pow10(R)
            intrem = 0
        else:
            c, e = None, None
        if c is None or e is None:
            # look for high enough precision to fit in the envelope
            prec = 100
            outside_envelope = True
            while outside_envelope:
                c, e, realrem = real_to_pow10_rem(R, prec)
                R_approx = pow10_to_real(c, e)
                if ((lower < R_approx and R_approx < upper) or
                    (lower_inclusive and lower == R_approx) or
                    (upper_inclusive and R_approx == upper)):
                    outside_envelope = False
                    if realrem.iszero:
                        intrem = 0
                    else:
                        intrem = realrem.sign
                else:
                    prec = prec * 2

        return binsearch_shortest_dec(c, e, intrem,
                                      lower, lower_inclusive, upper, upper_inclusive,
                                      round_correctly=round_correctly)

def implicit_to_shortest_dec(S, E, T, rm):
    assert isinstance(S, BV)
    assert S.n == 1
    assert isinstance(E, BV)
    assert E.n >= 2
    assert isinstance(T, BV)
    assert rm == core.RTN or rm == core.RTP or rm == core.RTZ or rm == core.RNE or rm == core.RNA

    R = core.implicit_to_real(S, E, T)

    return shortest_dec(R, S, E, T, rm)

def real_to_shortest_dec(R, w, p, rm, round_correctly = True):
    assert isinstance(R, FReal)
    assert isinstance(w, int)
    assert w >= 2
    assert isinstance(p, int)
    assert p >= 2
    assert rm == core.RTN or rm == core.RTP or rm == core.RTZ or rm == core.RNE or rm == core.RNA
    assert round_correctly is True or round_correctly is False

    S, E, T = core.real_to_implicit(R, w, p, rm)

    return shortest_dec(R, S, E, T, rm, round_correctly=round_correctly)

# canonical w and p for an ieee binary representation
def ieee_split_w_p(k):
    assert isinstance(k, int)

    if k == 16:
        return 5, 11
    elif k == 32:
        return 8, 24
    elif k == 64:
        return 11, 53
    elif k >= 128 and k % 32 == 0:
        w_offset_r = 4 * sympy.log(k, 2)
        w_offset_f = w_offset_r.evalf(20, maxn=real.default_maxn)

        if not w_offset_f.is_comparable:
            return None, None
        elif w_offset_f > sympy.Float('1e19', 20):
            return None, None

        w_offset_floor = int(sympy.floor(w_offset_f))

        realrem = (w_offset_r - w_offset_floor) - sympy.Rational(1,2)
        signrem = realrem.evalf(2, maxn=real.default_maxn)

        if not signrem.is_comparable:
            return None, None

        if signrem.is_positive:
            w_offset = w_offset_floor + 1
        else:
            w_offset = w_offset_floor

        w = w_offset - 13
        p = k - w

        return w, p
    else:
        return None, None

# see http://www.exploringbinary.com/number-of-digits-required-for-round-trip-conversions/
def bdb_round_trip_prec(p):
    assert isinstance(p, int)
    assert p >= 2

    r = p * sympy.log(2, 10)
    r_f = r.evalf(20, maxn=real.default_maxn)

    if not r_f.is_comparable:
        return None
    elif r_f > sympy.Float('1e19', 20):
        return None

    return int(sympy.ceiling(r_f)) + 1

# see http://www.exploringbinary.com/maximum-number-of-decimal-digits-in-binary-floating-point-numbers/
# this is an alternative (slow) formulation, not the logarithm formula
def dec_full_prec(w, p):
    assert isinstance(w, int)
    assert w >= 2
    assert isinstance(p, int)
    assert p >= 2

    S = BV(0, 1)
    E = BV(1, w)
    T = BV(-1, p-1)
    R = core.implicit_to_real(S, E, T)
    c, e = real_to_pow10(R)

    return ndig(c)
