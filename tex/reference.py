# bitvectors

def bitmask(n):
    if n > 0:
        return (1 << n) - 1
    else:
        return (-1) << (-n)

class BV(object):
    # should think about if bitvectors can have 0 size, and what sizes are required for different
    # versions of the FP representation

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

def count_leading_zeros(bv):
    assert isinstance(bv, BV)

    return bv.clz
    

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
    e = max(uint(E) - emax, emin)
    c = uint(C)

    if e > emax and c != 0:
        return Dec('nan')
    elif e > emax and c == 0:
        return ((-1) ** s) * Dec('inf')
    else: # e <= emax
        return ((-1) ** s) * (2 ** Dec(e)) * (c * (2 ** Dec(1 - p)))

# Equation 5
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
    e_prime = uint(E) - emax # can't min here, or we don't know what is a denorm
    c_prime = uint(T)

    if e_prime > emax and c_prime != 0:
        return Dec('nan')
    elif e_prime > emax and c_prime == 0:
        return ((-1) ** s) * Dec('inf')
    elif emin <= e_prime and e_prime <= emax:
        return ((-1) ** s) * (2 ** Dec(e_prime)) * ((c_prime + (2 ** Dec(p - 1))) * (2 ** Dec(1 - p)))
    else: # e_prime < emin
        return ((-1) ** s) * (2 ** Dec(emin)) * (c_prime * (2 ** Dec(1 - p)))

def implicit_to_explicit(S, E, T):
    assert isinstance(S, BV)
    assert size(S) == 1
    assert isinstance(E, BV)
    assert isinstance(T, BV)

    w = size(E)
    
    # Equation 4 - also broken, as we need to preserve infinities
    if uint(E) == 0 or uint(E) == (2 ** w) - 1 and uint(T) == 0:
        C = concat(BV(0, 1), T)
    else: # uint(E) != 0 and not an infinity
        C = concat(BV(1, 1), T)

    return S, E, C

def canonicalize(S, E, C):
    assert isinstance(S, BV)
    assert size(S) == 1
    assert isinstance(E, BV)
    assert isinstance(C, BV)

    w = size(E)
    emax = (2 ** (w - 1)) - 1
    emin = 1 - emax
    
    e = max(uint(E) - emax, emin)
    c = uint(C)

    # note that clz is not a simple bitvector arithmetic operation
    z = count_leading_zeros(C)
    h = e - emin
    offset = min(z, h)
        
    if e > emax:
        return S, E, C
    elif c == 0:
        return S, BV(0, w), C
    elif h < z:
        return S, BV(0, w), C << offset
    else: # h >= z
        return S, BV(e - offset + emax, w), C << offset

def is_canonical(S, E, C):
    assert isinstance(S, BV)
    assert size(S) == 1
    assert isinstance(E, BV)
    assert isinstance(C, BV)
    
    w = size(E)
    p = size(C)
    emax = (2 ** (w - 1)) - 1
    emin = 1 - emax
    
    e = max(uint(E) - emax, emin)
    c = uint(C)

    return e > emax or e == emin or C[p-1] == 1
    
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

    # There is a horrible case where deleting the implicit bit from a NaN will
    # create an infinity if the remaining T is 0. To fix this, return some other
    # NaN, here one with T=0b10..0
    if uint(E_canonical) == (2 ** w) - 1 and uint(C_canonical) != 0 and uint(T) == 0:
        return S_canonical, E_canonical, BV(1 << (p - 2), p - 1)
    else:
        return S_canonical, E_canonical, T


def test_fp_identical(a, b):
    a1, a2, a3 = a
    b1, b2, b3 = b
    return a1 == b1 and a2 == b2 and a3 == b3

def test_same_real_value(a, b):
    if a.is_nan() and b.is_nan():
        return True
    else:
        return a == b

def test_canon(w, p, verbose = False):
    for s in range(2**1):
        for e_prime in range(2**w):
            for c in range(2**p):

                S = BV(s, 1)
                E = BV(e_prime, w)
                C = BV(c, p)
                R = real_explicit(S, E, C)

                Sc, Ec, Cc = canonicalize(S, E, C)
                assert is_canonical(Sc, Ec, Cc)
                Rc = real_explicit(S, E, C)
                
                if verbose:
                    print('    {:20} {:20}'.format(R, Rc))
                if R.is_nan() and Rc.is_nan():
                    if not (S == Sc and E == Ec and C == Cc):
                        print('broken nan!')
                        print(S, E, C)
                        print(Sc, Ec, Cc)
                elif R != Rc:
                    print('BUG!    {:20} {:20}'.format(R, Rc))
                    print(S, E, C)
                    print(Sc, Ec, Cc)

                else:                    
                    w = size(Ec)
                    p = size(Cc)
                    emax = (2 ** (w - 1)) - 1
                    emin = 1 - emax
                    
                    s = uint(Sc)
                    e = max(uint(Ec) - emax, emin)
                    c = uint(Cc)

                    if not (Cc[p-1] == 1 or e == emin):
                        if e > emax:
                            assert R.is_infinite()
                        else:
                            print('canonicalization failed')
                            print(S, E, C)
                            print(Sc, Ec, Cc)
                    elif verbose and (E != Ec or C != Cc):
                            print( '  succeeded')
                            print(' ', S, E, C)
                            print(' ', Sc, Ec, Cc)

def test_explicit_implicit(w, p, verbose = False):
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

                if verbose:
                    print('    {:20} {:20} {:20} {:20}'.format(R, Ri, Rc, R1))

                assert test_same_real_value(R, Ri)
                assert test_same_real_value(R, Rc)

                # we lose information, so we can't check this for NaN
                if not R.is_nan():
                    assert test_fp_identical((Sc, Ec, Cc), (S1, E1, C1))
