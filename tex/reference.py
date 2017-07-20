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
    
    def __init__(self, i, n):
        assert isinstance(i, int)
        assert isinstance(n, int)
        assert n > 0
        
        self.i = i & bitmask(n)
        self.n = n

    def __str__(self):
        return ('0b{:0' + str(self.n) + 'b}').format(self.i)

    def __repr__(self):
        return ('BV(0b{:0' + str(self.n) + 'b}, {:d})').format(self.i, self.n)

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
# For displaying Reals, we can probably get away with Python's decimal module, as all we really
# want to do in this code is print things out.

from decimal import Decimal as Dec

# Equation 1
def real1(s, e, c, b, p):
    assert isinstance(s, int)
    assert s == 0 or s == 1
    assert isinstance(e, int)
    assert isinstance(c, int)
    assert isinstance(b, int)
    assert isinstance(p, int)
    
    return ((-1) ** s) * (b ** Dec(e)) * (c * (b ** Dec(1 - p)))

# Equation 2
def real2(s, e, C):
    assert isinstance(s, int)
    assert s == 0 or s == 1
    assert isinstance(e, int)
    assert isinstance(C, BV)

    return ((-1) ** s) * (2 ** Dec(e)) * (uint(C) * (2 ** Dec(1 - size(C))))

# Equation 3
def real_explicit(S, E, C):
    assert isinstance(S, BV)
    assert size(S) == 1
    assert isinstance(E, BV)
    assert isinstance(C, BV)
    
    w = size(E)
    p = size(C)
    emax = (2 ** (w - 1)) - 1
    emin = 1 - emax
    s = uint(S)
    e = uint(E) - emax
    c = uint(C)

    if e > emax and c != 0:
        return Dec('nan')
    elif e > emax and c == 0:
        return ((-1) ** s) * Dec('inf')
    elif emin <= e and e <= emax:
        return ((-1) ** s) * (2 ** Dec(e)) * (c * (2 ** Dec(1 - p)))
    else: # e < emin
        return ((-1) ** s) * (2 ** Dec(emin)) * (c * (2 ** Dec(1 - p)))


# Equation 4
def real_implicit(S, E, T):
    assert isinstance(S, BV)
    assert size(S) == 1
    assert isinstance(E, BV)
    assert isinstance(T, BV)

    w = size(E)
    p = size(T) + 1
    emax = (2 ** (w - 1)) - 1
    emin = 1 - emax
    s = uint(S)
    e_prime = uint(E) - emax
    c_prime = uint(T)

    if e_prime > emax and c_prime != 0:
        return Dec('nan')
    elif e_prime > emax and c_prime == 0:
        return ((-1) ** s) * Dec('inf')
    elif emin <= e_prime and e_prime <= emax:
        return ((-1) ** s) * (2 ** Dec(e_prime)) * ((c_prime + (2 ** Dec(p - 1))) * (2 ** Dec(1 - p)))
    else: # e_prime < emin
        return ((-1) ** s) * (2 ** Dec(emin)) * (c_prime * (2 ** Dec(1 - p)))

# Equation 5
def implicit_to_explicit(S, E, T):
    assert isinstance(S, BV)
    assert size(S) == 1
    assert isinstance(E, BV)
    assert isinstance(T, BV)

    w = size(E)
    
    if uint(E) == 0 or uint(E) == (2 ** w) - 1:
        C = concat(BV(0, 1), T)
    else: # uint(E) != 0 and not an infinity
        C = concat(BV(1, 1), T)

    return S, E, C

# Equation 6
def canonicalize(S, E, C):
    assert isinstance(S, BV)
    assert size(S) == 1
    assert isinstance(E, BV)
    assert isinstance(C, BV)

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
    assert isinstance(C, BV)
    
    w = size(E)
    p = size(C)
    emax = (2 ** (w - 1)) - 1
    emin = 1 - emax
    
    e_prime = max(uint(E) - emax, emin)
    c = uint(C)

    return e_prime > emax or e_prime == emin or C[p-1] == 1

# Equation 7
def explicit_to_implicit(S, E, C):
    assert isinstance(S, BV)
    assert size(S) == 1
    assert isinstance(E, BV)
    assert isinstance(C, BV)

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


# Sanity tests.

def test_fp_identical(a, b):
    a1, a2, a3 = a
    b1, b2, b3 = b
    return a1 == b1 and a2 == b2 and a3 == b3

def test_same_real_value(a, b):
    if a.is_nan() and b.is_nan():
        return True
    else:
        return a == b

def test_explicit_implicit(w, p, verbose = False, dots = 50):
    total = (2**1) * (2**w) * (2**p)
    if dots:
        dotmod = max(total // dots, 1)
    tested = 0
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

    for s in range(2**1):
        for e_prime in range(2**w):
            for c in range(2**p):

                S = BV(s, 1)
                E = BV(e_prime, w)
                C = BV(c, p)
                R = real_explicit(S, E, C)    
                            
                Si, Ei, Ti = explicit_to_implicit(S, E, C)
                Ri = real_implicit(Si, Ei, Ti)

                Sc, Ec, Cc = canonicalize(S, E, C)
                Rc = real_explicit(Sc, Ec, Cc)
                
                S1, E1, C1 = implicit_to_explicit(Si, Ei, Ti)
                R1 = real_explicit(S1, E1, C1)

                # print(' ', S, E, C)
                # print(' ', Si, Ei, Ti)
                # print(' ', Sc, Ec, Cc)
                # print(' ', S1, E1, C1)
                
                if verbose:
                    print('    {:20} {:20} {:20} {:20}'.format(R, Ri, Rc, R1))

                # These should always be canonical.
                assert is_canonical(Sc, Ec, Cc)
                assert is_canonical(S1, E1, C1)
                    
                # All real values should agree.
                assert test_same_real_value(R, Ri)
                assert test_same_real_value(R, Rc)
                assert test_same_real_value(R, R1)

                # We actually lose information when converting some NaNs to implicit,
                # so we can't get back the same canonical representation for them. We
                # should be able to do so for all other numbers.
                if not R.is_nan():
                    assert test_fp_identical((Sc, Ec, Cc), (S1, E1, C1))

                icov[(uint(Si), uint(Ei), uint(Ti),)] = True

                tested += 1
                if (not verbose) and dots and tested % dotmod == 0:
                    print('.', end='', flush=True)
    if (not verbose) and dots:
        print()

    # Check cover of implicit values
    assert all(icov.values())

    print('Tested {:d} explicit, {:d} implicit. Done.'.format(tested, len(icov)))
