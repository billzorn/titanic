import sys

# bitvectors

def bitmask(n):
    if n > 0:
        return (1 << n) - 1
    else:
        return (-1) << (-n)

class BV(object):
    # Bitvectors must have a size of at least 1.

    # test high (sign) bit
    def _negative(self):
        return self.i & (1 << (self.n - 1)) != 0
    negative = property(_negative)

    # unsigned integer representation
    def _uint(self):
        return self.i
    uint = property(_uint)

    # signed 2s complement integer representation
    def _sint(self):
        if self.negative:
            return self.i | bitmask(-n)
        else:
            return self.i
    sint = property(_sint)

    # count leading zeros; simple iterative implementation
    def _clz(self):
        if self.i == 0:
            return self.n
        i = self.i
        z = 0
        msb_mask = 1 << (self.n - 1)
        while i & msb_mask == 0:
            i = i << 1
            z = z + 1
        return z
    clz = property(_clz)

    def __init__(self, i, n = sys.byteorder):
        if isinstance(i, bytes):
            _n = len(i) * 8
            i = int.from_bytes(i, byteorder=n)
            n = _n

        assert isinstance(i, int)
        assert isinstance(n, int)
        assert n > 0

        self.i = i & bitmask(n)
        self.n = n

    def __str__(self):
        return ('0b{:0' + str(self.n) + 'b}').format(self.i)

    def __repr__(self):
        return ('BV(0b{:0' + str(self.n) + 'b}, {:d})').format(self.i, self.n)

    def to_bytes(self, byteorder=sys.byteorder):
        length = self.n // 8
        if self.n % 8 > 0:
            length += 1
        return int.to_bytes(self.i, length, byteorder=byteorder)

    # limited support for comparison

    def __eq__(self, y):
        assert isinstance(y, BV)
        assert y.n == self.n

        return self.i == y.i

    def __ne__(self, y):
        return not (self == y)

    # limited support for bitvector arithmetic operations

    def __lshift__(self, y):
        assert isinstance(y, int)
        assert y >= 0

        return BV((self.i << y), self.n)

    # this is arithmetic right shift
    def __rshift__(self, y):
        assert isinstance(y, int)
        assert y >= 0

        if self.negative:
            return BV((self.i >> y) | bitmask(-y), self.n)
        else:
            return BV(self.i >> y, self.n)

    # get the ith bit
    def __getitem__(self, k):
        assert(isinstance(k, int))
        assert 0 <= k and k < self.n

        return (self.i >> k) & 1


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


# Python's integers are a good stand-in for mathematical integers, as they can have arbitrary size.
# To represent reals, we can get away with rational numbers as represented by python's fractions,
# unless we want to do operations that might have irrational answers. In that case we would need
# constructive reals.

import math
def is_nan(r):
    return math.isnan(r)
def is_inf(r):
    return math.isinf(r)

import fractions
def Real(v):
    if v == 'nan':
        return float('nan')
    elif v == 'inf':
        return float('inf')
    elif is_nan(v) or is_inf(v):
        return v
    else:
        return fractions.Fraction(v)

# For printing out pretty decimal values. Optional sign parameter to print
# the correct sign for -0.
import decimal
def dec(r, prec, sign = None):
    if is_nan(r):
        return 'nan'
    elif is_inf(r):
        if r > 0:
            return 'inf'
        else:
            return '-inf'

    decimal.getcontext().prec = prec
    d = decimal.Decimal(r.numerator) / decimal.Decimal(r.denominator)

    if sign is not None and d == 0 and uint(sign) == 1:
        return '-' + str(d)
    else:
        return str(d)


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
def real_explicit(S, E, C):
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
def real_implicit(S, E, T):
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
    e_prime = uint(E) - emax
    c_prime = uint(T)

    if e_prime > emax and c_prime != 0:
        return Real('nan')
    elif e_prime > emax and c_prime == 0:
        return (Real(-1) ** s) * Real('inf')
    elif emin <= e_prime and e_prime <= emax:
        return (Real(-1) ** s) * (Real(2) ** e_prime) * ((Real(c_prime) + (Real(2) ** (p - 1))) * (Real(2) ** (1 - p)))
    else: # e_prime < emin
        return (Real(-1) ** s) * (Real(2) ** emin) * (Real(c_prime) * (Real(2) ** (1 - p)))

# Equation 5
def packf(S, E, T):
    assert isinstance(S, BV)
    assert size(S) == 1
    assert isinstance(E, BV)
    assert size(E) >= 2
    assert isinstance(T, BV)

    return concat(S, concat(E, T))

# Equation 6
def unpackf(B, w, p):
    assert isinstance(B, BV)
    assert w >= 2
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
def ordinal(S, E, T):
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
def refloat(i, w, p):
    assert w >= 2
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
def ordinal_packed(B, w, p):
    assert isinstance(B, BV)
    assert w >= 2
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
def refloat_packed(i, w, p):
    assert w >= 2
    assert p >= 2

    umax = ((2 ** w) - 1) * (2 ** (p - 1))

    assert -umax <= i and i <= umax

    u = abs(i)
    ET = BV(u, w + p - 1)

    if i >= 0:
        return concat(BV(0, 1), ET)
    else: # i < 0
        return concat(BV(1, 1), ET)


def binsearch_nearest_ord(frac, w, p):
    assert isinstance(frac, fractions.Fraction)
    assert w >= 2
    assert p >= 2

    umax = ((2 ** w) - 1) * (2 ** (p - 1))

    below = -umax
    above = umax

    while above - below > 1:
        between = below + ((above - below) // 2)
        S, E, T = refloat(between, w, p)
        guess = real_implicit(S, E, T)

        if guess > frac:
            above = between
        elif guess < frac:
            below = between
        else: # exact equality, return
            assert guess == frac
            return between, between

    assert above - below == 1
    return below, above

# Floating point rounding
def round_to_float(frac, below, above, w, p, rm):
    assert isinstance(frac, fractions.Fraction)
    assert w >= 2
    assert p >= 2
    umax = ((2 ** w) - 1) * (2 ** (p - 1))
    assert -umax <= below and below <= above and above <= umax
    assert above - below <= 1
    assert rm == 'RTP' or rm == 'RTN' or rm == 'RTZ' or rm == 'RNE' or rm == 'RNA'

    # prec = max(28, 2 ** w, p * 2)
    # Sa, Ea, Ta = refloat(above, w, p)
    # Sb, Eb, Tb = refloat(below, w, p)
    # Ra = real_implicit(Sa, Ea, Ta)
    # Rb = real_implicit(Sb, Eb, Tb)
    # Da = dec(Ra, prec, sign=Sa)
    # Df = dec(frac, prec)
    # Db = dec(Rb, prec, sign=Sb)

    # print('  above: ', above, Sa, Ea, Ta, Ra, Da)
    # print('  frac : ', frac, Df)
    # print('  below: ', below, Sb, Eb, Tb, Rb, Db)

    # exact equality
    if below == above:
        final = below
    elif rm == 'RTP':
        # print('RTP, final = above')
        final = above
    elif rm == 'RTN':
        # print('RTN, final = below')
        final = below
    elif rm == 'RTZ':
        if above > 0:
            # print('RTZ, final = below')
            final = below
        else:
            # print('RTZ, final = above')
            final = above
    else: # rm == 'RNE' or rm == 'RNA'
        emax = (2 ** (w - 1)) - 1
        fmax = (Real(2) ** emax) * (Real(2) - ((Real(2) ** (1 - p)) / Real(2)))

        if frac >= fmax:
            # print('RN, final = umax')
            final = umax
        elif frac <= -fmax:
            # print('RN, final = -umax')
            final = -umax
        else:
            Sa, Ea, Ta = refloat(above, w, p)
            Sb, Eb, Tb = refloat(below, w, p)
            guess_above = real_implicit(Sa, Ea, Ta)
            guess_below = real_implicit(Sb, Eb, Tb)

            assert guess_below < frac and frac < guess_above

            difference_above = guess_above - frac
            difference_below = frac - guess_below

            # print('    ^', difference_above, dec(difference_above, prec))
            # print('    v', difference_below, dec(difference_below, prec))

            if difference_above < difference_below:
                # print('RN, final = above')
                final = above
            elif difference_above > difference_below:
                # print('RN, final = below')
                final = below
            else: # exact equality, find even or away
                assert difference_above == difference_below
                if rm == 'RNE':
                    if Ta[0] == 0:
                        # print('RNE, final = above')
                        final = above
                    else:
                        assert Tb[0] == 0
                        # print('RNE, final = below')
                        final = below
                else: # rm == 'RNA'
                    if above > 0:
                        # print('RNA, final = above')
                        final = above
                    else:
                        # print('RNA, final = below')
                        final = below

    return refloat(final, w, p)

def str_to_implicit(s, w, p, rm = 'RNE'):

    # inf and nan behavior modeled on numpy
    sl = s.strip().lower()
    if sl == 'inf' or sl == '+inf' or sl == 'infinity' or sl == '+infinity':
        return BV(0, 1), BV(-1, w), BV(0, p-1)
    elif sl == '-inf' or sl == '-infinity':
        return BV(1, 1), BV(-1, w), BV(0, p-1)
    elif sl == 'nan' or sl == '+nan':
        return BV(0, 1), BV(-1, w), BV(2 ** (p-2), p-1)
    elif sl == '-nan':
        return BV(1, 1), BV(-1, w), BV(2 ** (p-2), p-1)
    else:
        frac = fractions.Fraction(s)
        # special case for -0, which doesn't exist in fractions...
        if frac == 0 and sl.startswith('-'):
            return BV(1, 1), BV(0, w), BV(0, p-1)
        else:
            below, above = binsearch_nearest_ord(frac, w, p)
            return round_to_float(frac, below, above, w, p, rm)


# Conversions for numpy 16, 32, and 64-bit floats.
# float16 : w = 5,  p = 11
# float32 : w = 8,  p = 24
# float64 : w = 11, p = 53
# float128: not an IEEE 754 128-bit float, possibly 80bit x87?
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
    return unpackf(B, w, p)

def packed_to_np_float(B):
    assert size(B) == 16 or size(B) == 32 or size(B) == 64

    if size(B) == 16:
        ftype = np.float16
    elif size(B) == 32:
        ftype = np.float32
    else: # size(B) == 64
        ftype = np.float64

    return np.frombuffer(B.to_bytes(byteorder=np_byteorder(ftype)), dtype=ftype, count=1, offset=0)[0]

def implicit_to_np_float(S, E, T):
    assert isinstance(S, BV)
    assert size(S) == 1
    assert isinstance(E, BV)
    assert isinstance(T, BV)
    assert (size(E) == 5 and size(T) == 10) or (size(E) == 8 and size(T) == 23) or (size(E) == 11 and size(T) == 52)

    return packed_to_np_float(packf(S, E, T))


# Sanity tests.

def test_fp_identical(a, b):
    a1, a2, a3 = a
    b1, b2, b3 = b
    return a1 == b1 and a2 == b2 and a3 == b3

def test_same_real_value(a, b):
    if is_nan(a) and is_nan(b):
        return True
    else:
        return a == b

def test_explicit_implicit(w, p, verbose = False, dots = 50):
    total = (2**1) * (2**w) * (2**p)
    umax = ((2 ** w) - 1) * (2 ** (p - 1))
    if dots:
        dotmod = max(total // dots, 1)
    tested = 0
    # precision to use when printing in decimal form... not sure if this is enough
    prec = max(28, 2 ** w, p * 2)

    print('{:d} explicit representations to test.'.format(total))

    # Make sure all possible implicit values are covered.
    icov = {}
    for s in range(2**1):
        for e_prime in range(2**w):
            for c_prime in range(2**(p-1)):
                S = BV(s, 1)
                E = BV(e_prime, w)
                T = BV(c_prime, p-1)

                icov[(uint(S), uint(E), uint(T),)] = False

    # Make sure all possible ordinals are covered.
    ocov = {i : False for i in range(-umax, umax + 1)}

    # Make sure all packed bitvectors are covered.
    pcov = {x : False for x in range((2 ** w) * (2 ** p))}

    # Test
    for s in range(2**1):
        for e_prime in range(2**w):
            for c in range(2**p):

                S = BV(s, 1)
                E = BV(e_prime, w)
                C = BV(c, p)
                R = real_explicit(S, E, C)

                Si, Ei, Ti = explicit_to_implicit(S, E, C)
                Ri = real_implicit(Si, Ei, Ti)
                try:
                    i = ordinal(Si, Ei, Ti)
                    So, Eo, To = refloat(i, w, p)
                    Ro = real_implicit(So, Eo, To)
                    inrange = True
                except OrdinalError:
                    i = 'undefined'
                    inrange = False

                B = packf(Si, Ei, Ti)
                Sp, Ep, Tp = unpackf(B, w, p)

                try:
                    ip = ordinal_packed(B, w, p)
                    Bo = refloat_packed(ip, w, p)
                    inrange_packed = True
                except OrdinalError:
                    ip = 'packord out of range'
                    inrange_packed = False

                Sc, Ec, Cc = canonicalize(S, E, C)
                Rc = real_explicit(Sc, Ec, Cc)

                S1, E1, C1 = implicit_to_explicit(Si, Ei, Ti)
                R1 = real_explicit(S1, E1, C1)

                if verbose:
                    print('  {} {} {} {}  {:10} {:20} {:20}'
                          .format(Si, Ei, C[p-1], Ti, str(i), str(R), dec(R, prec, sign=S)))

                # These should always be canonical.
                assert is_canonical(Sc, Ec, Cc)
                assert is_canonical(S1, E1, C1)

                # All real values should agree.
                assert test_same_real_value(R, Ri)
                if inrange:
                    assert test_same_real_value(R, Ro)
                assert test_same_real_value(R, Rc)
                assert test_same_real_value(R, R1)

                # We actually lose information when converting some NaNs to implicit,
                # so we can't get back the same canonical representation for them. We
                # should be able to do so for all other numbers.
                if not is_nan(R):
                    assert test_fp_identical((Sc, Ec, Cc,), (S1, E1, C1,))

                # We should also get identical representations back from ordinals, except
                # for the zeros which both map to i=0.
                if inrange and R != 0:
                    assert test_fp_identical((Si, Ei, Ti,), (So, Eo, To,))

                # Packed representations should always give us back the same thing.
                assert test_fp_identical((Si, Ei, Ti,), (Sp, Ep, Tp,))

                # Ordinals should work the same with packed and unpacked implicit.
                assert inrange == inrange_packed
                if inrange:
                    assert i == ip
                    if R != 0:
                        assert B == Bo

                icov[(uint(Si), uint(Ei), uint(Ti),)] = True
                if inrange:
                    ocov[i] = True
                pcov[uint(B)] = True

                tested += 1
                if (not verbose) and dots and tested % dotmod == 0:
                    print('.', end='', flush=True)
    if (not verbose) and dots:
        print()

    # Check cover of implicit values, ordinals, and packed representations.
    assert all(icov.values())
    assert all(ocov.values())
    assert all(pcov.values())

    print('Tested {:d} explicit, {:d} implicit, {:d} ordinals. Done.'.format(tested, len(icov), len(ocov)))

def test_numpy_fp(points, ftype, verbose = False, dots = 50):
    total = len(points)
    if dots:
        dotmod = max(total // dots, 1)
    tested = 0

    print('{:d} points to test against {}.'.format(total, ftype))

    # We could be more conservative with the printed precision, but I don't trust numpy
    # any farther than I can throw it. Minimum precision that seems to work is in comments.

    if ftype == np.float16:
        width = 16
        w = 5
        p = 11
        #fmt = '{:1.4e}'
        fmt = '{:1.5e}'
    elif ftype == np.float32:
        width = 32
        w = 8
        p = 24
        #fmt = '{:1.8e}'
        fmt = '{:1.9e}'
    elif ftype == np.float64:
        width = 64
        w = 11
        p = 53
        #fmt = '{:1.17e}'
        fmt = '{:1.18e}'
    else:
        raise ValueError('unsupported floating point format {}'.format(repr(ftype)))

    rm = 'RNE'

    for x in points:
        B = BV(x, width)
        f = packed_to_np_float(B)
        S, E, T = np_float_to_implicit(f)
        B1 = packf(S, E, T)
        f1 = implicit_to_np_float(S, E, T)

        R = real_implicit(S, E, T)
        # This only works because python's floats have at least as much precision
        # as the supported numpy types.
        Rf = Real(float(f))

        s = fmt.format(f)
        Ss, Es, Ts = str_to_implicit(s, w, p, rm)
        Rs = real_implicit(Ss, Es, Ts)

        if verbose:
            try:
                i = ordinal(S, E, T)
            except OrdinalError:
                i = 'ooor'
            print('we wanted: {}'.format(i))
            print(S, E, T, R, s, Ss, Es, Ts, Rs)
            print()

        assert B == B1

        if is_nan(R):
            assert is_nan(Rf)
            assert is_nan(f)
            assert is_nan(f1)
            assert is_nan(Rs)
        else:
            assert B == B1
            assert R == Rf
            assert f == f1
            # This depends on the formatting, if it fails make sure
            # you asked for enough digits and got the right string.
            assert test_fp_identical((S, E, T,), (Ss, Es, Ts,))

        tested += 1

        if (not verbose) and dots and tested % dotmod == 0:
            print('.', end='', flush=True)
    if (not verbose) and dots:
        print()

    print('Tested {:d} points against {}. Done.'.format(tested, ftype))


# Automatic tests
if __name__ == '__main__':
    import random

    test_explicit_implicit(2, 2, True)

    test_explicit_implicit(4, 8, False, dots=8)
    test_explicit_implicit(5, 11, False, dots=131)
    test_explicit_implicit(11, 5, False, dots=131)

    test_numpy_fp(range(2**16), np.float16, False, dots=65)

    test_numpy_fp([random.randrange(2**32) for j in range(100000)], np.float32, False, dots=100)
    test_numpy_fp([random.randrange(2**64) for j in range(100000)], np.float64, False, dots=100)
