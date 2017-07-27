import sys

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
