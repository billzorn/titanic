from bv import BV
from real import Real

# bitvector operations, as written out in the paper

def uint(bv):
    assert isinstance(bv, BV)

    return bv.uint

def size(bv):
    assert isinstance(bv, BV)

    return bv.n

def clz(bv):
    assert isinstance(bv, BV)

    return bv.clz

def concat(bv1, bv2):
    assert isinstance(bv1, BV)
    assert isinstance(bv2, BV)

    return BV((bv1.i << bv2.n) | bv2.i, bv1.n + bv2.n)

# bounds are inclusive, first argument is higher index, higher indices are more significant bits
def extract(left, right, bv):
    assert isinstance(left, int)
    assert isinstance(right, int)
    assert left >= right and right >= 0
    assert left < bv.n

    return BV((bv.i >> right), (left - right) + 1)

# core equations from the paper

# Equation 1
def real1(s, e, c, b, p):
    assert isinstance(s, int)
    assert s == 0 or s == 1
    assert isinstance(e, int)
    assert isinstance(c, int)
    assert c >= 0
    assert isinstance(b, int)
    assert b >= 2
    assert isinstance(p, int)
    assert p >= 1

    return (Real(-1) ** s) * (Real(b) ** e) * (Real(c) * (Real(b) ** (1 - p)))

# Equation 2
def real2(s, e, C):
    assert isinstance(s, int)
    assert s == 0 or s == 1
    assert isinstance(e, int)
    assert isinstance(C, BV)

    return (Real(-1) ** s) * (Real(2) ** e) * (Real(uint(C)) * (Real(2) ** (1 - size(C))))

# Equation 3
def explicit_to_real(S, E, C):
    assert isinstance(S, BV)
    assert size(S) == 1
    assert isinstance(E, BV)
    assert size(E) >= 2
    assert isinstance(C, BV)
    assert size(C) >= 2

    w = size(E)
    p = size(C)
    emax = (2 ** (w - 1)) - 1
    emin = 1 - emax
    s = uint(S)
    e = uint(E) - emax
    c = uint(C)

    if e > emax and c != 0:
        return Real('nan')
    elif e > emax and c == 0:
        return (Real(-1) ** s) * Real('inf')
    elif emin <= e and e <= emax:
        return (Real(-1) ** s) * (Real(2) ** e) * (Real(c) * (Real(2) ** (1 - p)))
    else: # e < emin
        return (Real(-1) ** s) * (Real(2) ** emin) * (Real(c) * (Real(2) ** (1 - p)))

# Equation 4
def implicit_to_real(S, E, T):
    assert isinstance(S, BV)
    assert size(S) == 1
    assert isinstance(E, BV)
    assert size(E) >= 2
    assert isinstance(T, BV)

    w = size(E)
    p = size(T) + 1
    emax = (2 ** (w - 1)) - 1
    emin = 1 - emax
    s = uint(S)
    e = uint(E) - emax
    c_prime = uint(T)

    if e > emax and c_prime != 0:
        return Real('nan')
    elif e > emax and c_prime == 0:
        return (Real(-1) ** s) * Real('inf')
    elif emin <= e and e <= emax:
        return (Real(-1) ** s) * (Real(2) ** e) * ((Real(c_prime) + (Real(2) ** (p - 1))) * (Real(2) ** (1 - p)))
    else: # e < emin
        return (Real(-1) ** s) * (Real(2) ** emin) * (Real(c_prime) * (Real(2) ** (1 - p)))

# Equation 5
def implicit_to_packed(S, E, T):
    assert isinstance(S, BV)
    assert size(S) == 1
    assert isinstance(E, BV)
    assert size(E) >= 2
    assert isinstance(T, BV)

    return concat(S, concat(E, T))

# Equation 6
def packed_to_implicit(B, w, p):
    assert isinstance(B, BV)
    assert isinstance(w, int)
    assert w >= 2
    assert isinstance(p, int)
    assert p >= 2
    assert size(B) == w + p

    return extract(w + p - 1, w + p - 1, B), extract(w + p - 2, p - 1, B), extract(p - 2, 0, B)

# Equation 7
def implicit_to_explicit(S, E, T):
    assert isinstance(S, BV)
    assert size(S) == 1
    assert isinstance(E, BV)
    assert size(E) >= 2
    assert isinstance(T, BV)

    w = size(E)

    if uint(E) == 0 or uint(E) == (2 ** w) - 1:
        C = concat(BV(0, 1), T)
    else: # uint(E) != 0 and not an infinity
        C = concat(BV(1, 1), T)

    return S, E, C

# Equation 8
def canonicalize(S, E, C):
    assert isinstance(S, BV)
    assert size(S) == 1
    assert isinstance(E, BV)
    assert size(E) >= 2
    assert isinstance(C, BV)
    assert size(C) >= 2

    w = size(E)
    emax = (2 ** (w - 1)) - 1
    emin = 1 - emax

    e_prime = max(uint(E) - emax, emin)
    c = uint(C)

    # Note that clz is not a simple bitvector arithmetic operation
    z = clz(C)
    h = e_prime - emin
    x = min(z, h)

    if e_prime > emax:
        return S, E, C
    elif c == 0:
        return S, BV(0, w), C
    elif h < z:
        return S, BV(0, w), C << x
    else: # h >= z
        return S, BV(e_prime - x + emax, w), C << x

def is_canonical(S, E, C):
    assert isinstance(S, BV)
    assert size(S) == 1
    assert isinstance(E, BV)
    assert size(E) >= 2
    assert isinstance(C, BV)
    assert size(C) >= 2

    w = size(E)
    p = size(C)
    emax = (2 ** (w - 1)) - 1
    emin = 1 - emax

    e_prime = max(uint(E) - emax, emin)
    c = uint(C)

    return e_prime > emax or e_prime == emin or C[p-1] == 1

# Equation 9
def explicit_to_implicit(S, E, C):
    assert isinstance(S, BV)
    assert size(S) == 1
    assert isinstance(E, BV)
    assert size(E) >= 2
    assert isinstance(C, BV)
    assert size(C) >= 2

    S_canonical, E_canonical, C_canonical = canonicalize(S, E, C)
    assert is_canonical(S_canonical, E_canonical, C_canonical)

    w = size(E_canonical)
    p = size(C_canonical)
    T = extract(p - 2, 0, C_canonical)

    # There is an edge case where deleting the implicit bit from a NaN will
    # create an infinity if the remaining T is 0. To fix this, return some other
    # NaN instead, here one with T=0b10..0

    if uint(E_canonical) == (2 ** w) - 1 and uint(C_canonical) != 0 and uint(T) == 0:
        return S_canonical, E_canonical, BV(2 ** (p - 2), p - 1)
    else:
        return S_canonical, E_canonical, T

class OrdinalError(ArithmeticError):
    pass

# Equation 10
def implicit_to_ordinal(S, E, T):
    assert isinstance(S, BV)
    assert size(S) == 1
    assert isinstance(E, BV)
    assert size(E) >= 2
    assert isinstance(T, BV)

    w = size(E)
    p = size(T) + 1
    umax = ((2 ** w) - 1) * (2 ** (p - 1))
    s = uint(S)
    u = (uint(E) * (2 ** (p - 1))) + uint(T)

    # alternatively,
    # u = uint(concat(E,T))

    if u > umax:
        raise OrdinalError()
    else: # u <= umax
        return ((-1) ** s) * u

# Equation 11
def ordinal_to_implicit(i, w, p):
    assert isinstance(i, int)
    assert isinstance(w, int)
    assert w >= 2
    assert isinstance(p, int)
    assert p >= 2
    umax = ((2 ** w) - 1) * (2 ** (p - 1))
    assert -umax <= i and i <= umax

    u = abs(i)
    U = BV(u, w + p - 1)
    E = extract(w + p - 2, p - 1, U)
    T = extract(p - 2, 0, U)

    if i >= 0:
        return BV(0, 1), E, T
    else: # i < 0
        return BV(1, 1), E, T

# Equation 12
def packed_to_ordinal(B, w, p):
    assert isinstance(B, BV)
    assert isinstance(w, int)
    assert w >= 2
    assert isinstance(p, int)
    assert p >= 2
    assert size(B) == w + p

    S = extract(w + p - 1, w + p - 1, B)
    ET = extract(w + p - 2, 0, B)
    umax = ((2 ** w) - 1) * (2 ** (p - 1))
    s = uint(S)
    u = uint(ET)

    if u > umax:
        raise OrdinalError()
    else: # u <= umax
        return ((-1) ** s) * u

# Equation 13
def ordinal_to_packed(i, w, p):
    assert isinstance(i, int)
    assert isinstance(w, int)
    assert w >= 2
    assert isinstance(p, int)
    assert p >= 2
    umax = ((2 ** w) - 1) * (2 ** (p - 1))
    assert -umax <= i and i <= umax

    u = abs(i)
    ET = BV(u, w + p - 1)

    if i >= 0:
        return concat(BV(0, 1), ET)
    else: # i < 0
        return concat(BV(1, 1), ET)

# markers for IEEE 754 rounding modes
RTN = 'RTN' # roundTowardNegative
RTP = 'RTP' # roundTowardPositive
RTZ = 'RTZ' # roundTowardZero
RNE = 'RNE' # roundTiesToEven
RNA = 'RNA' # roundTiesToAway

# Find the two closest floating point numbers representable with w and p
# to some real. This works by performing a simple binary search over the
# set of ordinals. The two closest numbers are returned as ordinals. Iff
# an exact equality is found, the two ordinals will be the same. The real
# number may be infinite, but not NaN.
def binsearch_nearest_ordinals(r, w, p):
    assert isinstance(r, Real)
    assert not r.isnan
    assert isinstance(w, int)
    assert w >= 2
    assert isinstance(p, int)
    assert p >= 2

    umax = ((2 ** w) - 1) * (2 ** (p - 1))

    # First deal with pesky infinities.
    if r.isinf:
        if r > 0:
            return umax, umax
        else: # r < 0
            return -umax, -umax

    below = -umax
    above = umax

    while above - below > 1:
        between = below + ((above - below) // 2)
        S, E, T = ordinal_to_implicit(between, w, p)
        guess = implicit_to_real(S, E, T)

        if r < guess:
            above = between
        elif guess < r:
            below = between
        else: # exact equality, return
            assert guess == r
            return between, between

    # We can't possibly have either bound be exact, since they were inexact to start
    # (we checked the infinities) and we only reassigned under strict inequalities.
    assert above - below == 1
    return below, above

# Correct IEEE 754 rounding, given the two nearest ordinals.
def ieee_round_to_implicit(r, below, above, w, p, rm):
    assert isinstance(r, Real)
    assert not r.isnan
    assert isinstance(below, int)
    assert isinstance(above, int)
    assert isinstance(w, int)
    assert w >= 2
    assert isinstance(p, int)
    assert p >= 2
    umax = ((2 ** w) - 1) * (2 ** (p - 1))
    assert -umax <= below and below <= above and above <= umax
    assert above - below <= 1
    assert rm == RTN or rm == RTP or rm == RTZ or rm == RNE or rm == RNA

    # exact equality
    if below == above:
        final = below

    # otherwise, we have to round
    elif rm == RTN:
        final = below
    elif rm == RTP:
        final = above
    elif rm == RTZ:
        if above > 0:
            final = below
        else:
            final = above

    else: # rm == RNE or rm == RNA
        emax = (2 ** (w - 1)) - 1
        fmax = (Real(2) ** emax) * (Real(2) - ((Real(2) ** (1 - p)) / Real(2)))

        if r <= -fmax:
            final = -umax
        elif fmax <= r:
            final = umax
        else:
            Sb, Eb, Tb = ordinal_to_implicit(below, w, p)
            Sa, Ea, Ta = ordinal_to_implicit(above, w, p)
            guess_below = implicit_to_real(Sb, Eb, Tb)
            guess_above = implicit_to_real(Sa, Ea, Ta)

            assert guess_below < r and r < guess_above

            difference_below = r - guess_below
            difference_above = guess_above - r

            if difference_below < difference_above:
                final = below
            elif difference_above < difference_below:
                final = above
            else: # exactly halfway, round to even or away
                assert difference_above == difference_below
                if rm == RNE:
                    if Tb[0] == 0:
                        final = below
                    else:
                        assert Ta[0] == 0
                        final = above
                else: # rm == RNA
                    if below < 0:
                        final = below
                    else:
                        final = above

    return ordinal_to_implicit(final, w, p)

# Here, r can be nan, and we return some nan representation.
def real_to_implicit(r, w, p, rm):
    assert isinstance(r, Real)
    assert isinstance(w, int)
    assert w >= 2
    assert isinstance(p, int)
    assert p >= 2
    assert rm == RTN or rm == RTP or rm == RTZ or rm == RNE or rm == RNA

    if r.isnan:
        return BV(0, 1), BV(-1, w), BV(1, p-1)
    else:
        above, below = binsearch_nearest_ordinals(r, w, p)
        return ieee_round_to_implicit(r, above, below, w, p, rm)
