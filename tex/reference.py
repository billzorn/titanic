# bitvectors

def bitmask(n):
    if n > 0:
        return (1 << n) - 1
    else:
        return (-1) << (-n)

class BV(object):

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
    e = max(uint(E) - emax, emin)
    c_prime = uint(T)

    if e > emax and c != 0:
        return Dec('nan')
    elif e > emax and c == 0:
        return ((-1) ** s) * Dec('inf')
    elif emin < e and e <= emax:
        return ((-1) ** s) * (2 ** Dec(e)) * ((c_prime + (2 ** Dec(p - 1))) * (2 ** Dec(1 - p)))
    else: # e = emin
        return ((-1) ** s) * (2 ** Dec(e)) * (c_prime * (2 ** Dec(1 - p)))

def implicit_to_explicit(S, E, T):
    assert isinstance(S, BV)
    assert size(S) == 1
    assert isinstance(E, BV)
    assert isinstance(T, BV)

    # Equation 4
    if uint(E) == 0:
        C = concat(BV(0, 1), T)
    else: # uint(E) != 0
        C = concat(BV(1, 1), T)

    return S, E, C

def canonicalize(S, E, C):
    assert isinstance(S, BV)
    assert size(S) == 1
    assert isinstance(E, BV)
    assert isinstance(C, BV)

    
