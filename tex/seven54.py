def mask(n):
    return (2**n)-1

def mask_at(x, n, i):
    return (x >> i) & mask(n)

inf = float('inf')
nan = float('nan')

def fromordinal(x, ebits, sbits):
    assert sbits >= 1
    
    if x < 0:
        S = 1
        u = -x
    else:
        S = 0
        u = x

    eb = ebits
    sb = sbits - 1
    E = (u >> sb) & mask(eb)
    T = u & mask(sb)

    i = (S << (eb + sb)) | (E << sb) | T
    return F(i, ebits, sbits)

def fromordinal2(x, ebits, sbits):
    assert sbits >= 1

    if x < 0:
        s = 1
        u = -x
    else:
        s = 0
        u = x
    S = s

    w = ebits
    p = sbits
    bias = (2 ** (w - 1)) - 1
    e = ((u >> (p - 1)) & mask(w)) - bias
    E = e + bias

    if E == 0:
        c = (u & mask(p - 1))
    else:
        c = (u & mask(p - 1)) | (1 << (p - 1))
    T = c & mask(p - 1)

    i = (S << (w + (p - 1))) | (E << (p - 1)) | T
    return F(i, ebits, sbits)

def real3(S, E, C, w, p):
    emax = (2 ** (w - 1)) - 1
    emin = 1 - emax
    s = S & mask(1)
    e = max((E & mask(w)) - emax, emin)
    c = C & mask(p)

    if e > emax and c != 0:
        return float('nan')
    elif e > emax and c == 0:
        return ((-1) ** s) * float('inf')
    elif e <= emax:
        return ((-1) ** s) * (2 ** e) * (c * (2 ** (1 - p)))

def ord3_cheat(S, E, C, w, p):
    T = C & mask(p-1)
    umax = ((2 ** w) - 1) * (2 ** (p - 1))
    s = S & mask(1)
    u = ((E & mask(w)) * (2 ** (p - 1))) + (T & mask(w))

    if u > umax:
        return float('nan')
    else:
        return ((-1) ** s) * u
        

class F(object):
    def _get_k(self):
        return self.eb + self.sb + 1
    k = property(_get_k)

    def _get_w(self):
        return self.eb
    w = property(_get_w)

    def _get_p(self):
        return self.sb + 1
    p = property(_get_p)

    def _get_emax(self):
        return (2 ** (self.w - 1)) - 1
    emax = property(_get_emax)
    bias = property(_get_emax)

    def _get_emin(self):
        return 1 - self.emax
    emin = property(_get_emin)

    def _get_e(self):
        return self.E - self.bias
    e = property(_get_e)

    def _get_c(self):
        if self.E == 0:
            return self.T
        else:
            return self.T | (1 << self.sb)
    c = property(_get_c)
        
    def _get_m(self):
        unbiased = self.T * (2 ** (1 - self.p))
        if self.E == 0:
            return unbiased
        else:
            return 1 + unbiased
    def _get_m2(self):
        return self.c * (2 ** (1 - self.p))
    m = property(_get_m2)
        
    def __init__(self, i, ebits, sbits):
        self.i = i
        assert sbits >= 1
        self.eb = ebits
        self.sb = sbits - 1
        assert 0 <= i and i < 2 ** self.k

        self.S = mask_at(i, 1, self.eb + self.sb)
        self.E = mask_at(i, self.eb, self.sb)
        self.T = mask_at(i, self.sb, 0)

        # print(('{:0' + str(self.k) + 'b}').format(i))
        # print(('{:01b} {:0' + str(self.eb) + 'b} {:0' + str(self.sb) + 'b}').format(self.S, self.E, self.T))

    # not really, hahaha, but close enough for small precisions
    def real(self):
        # NaN, Inf
        if self.E == (2**self.w) - 1:
            if self.T == 0:
                return ((-1) ** self.S) * inf
            else:
                return nan
        # Subnormal
        elif self.E == 0:
            return ((-1) ** self.S) * (2 ** self.emin) * self.m
        # Normal
        else:
            assert self.emin <= self.e and self.e <= self.emax
            return ((-1) ** self.S) * (2 ** self.e) * self.m

    def _ordinal(self):
        u = (self.E << self.sb) | self.T
        if self.S == 0:
            return u
        else:
            return -u

    def _ordinal2(self):
        u = ((self.e + self.bias) << self.sb) | (self.c & mask(self.p - 1))
        if self.S == 0:
            return u
        else:
            return -u        

    def ordinal(self):
        if self.E == (2**self.w) - 1 and self.T != 0:
            return nan
        
        if not self._ordinal() == self._ordinal2():
            print(self._ordinal(), self._ordinal2())

        x = self._ordinal2()

        f = fromordinal(x, self.eb, self.sb + 1)
        f2 = fromordinal2(x, self.eb, self.sb + 1)

        x1 = f._ordinal()
        x2 = f2._ordinal2()

        if not x == x1:
            print('ordinal: expected {}, got {}'.format(x, x1))
        if not x == x2:
            print('ordinal2: expected {}, got {}'.format(x, x2))


        return self._ordinal()
        
        
    def __str__(self):
        return 'F({:#b}) = {} @ {}'.format(self.i, self.real(), self.ordinal())

    def __repr__(self):
        return str(self)

if __name__ == '__main__':
    import struct, math

    # for i in range(2**6):
    #     f = F(i, 3, 4)
    #     print(f)

    w = 3
    p = 3
    for S in range(2):
        for E in range(2**w):
            for C in range(2**p):
                print('{}\t{}'.format(real3(S, E, C, w, p), ord3_cheat(S, E, C, w, p)))

        
    # for i in range(2**20):
    #     x = F(i, 8, 24).real()
    #     y = struct.unpack('f', struct.pack('I', i))[0]

    #     if x != y:
    #         if not (math.isnan(x) and math.isnan(y)):
    #             print(x, y)

    # for i in range(2**4):
    #     f = F(i, 2, 2)
    #     fullord = f._ordinal()
    #     o = f.ordinal()
    #     r = f.real()

    #     if math.isinf(r):
    #         if f.S == 0:
    #             r = '\\infty'
    #         else:
    #             r = '-\\infty'
    #     elif math.isnan(r):
    #         r = 'nan'

    #     if math.isnan(o):
    #         o = 'nan'
                
    #     if fullord == 0:
    #         if f.S == 0:
    #             # report +- 0 for +0
    #             print('{}/{}/\\pm{},'.format(fullord, o, r), end='')
    #         # skip -0
    #     else:
    #         # otherwise print normally
    #         print('{}/{}/{},'.format(fullord, o, r), end='')
    # print()

    # for i in range(2**4):
    #     f = F(i, 2, 2)
    #     fullord = f._ordinal()
    #     o = f.ordinal()
    #     r = f.real()

    #     if math.isinf(r):
    #         if f.S == 0:
    #             r = '\\infty'
    #         else:
    #             r = '-\\infty'
    #     elif math.isnan(r):
    #         r = 'nan'

    #     if math.isnan(o):
    #         o = 'nan'
                
    #     if fullord == 0:
    #         if f.S == 0:
    #             # report +- 0 for +0
    #             print('{}/{}/\\pm{},'.format(r, o, r), end='')
    #         # skip -0
    #     else:
    #         # otherwise print normally
    #         print('{}/{}/{},'.format(r, o, r), end='')
    # print()
    

    # for i in range(2**6):
    #     f = F(i, 3, 3)
    #     fullord = f._ordinal()
    #     o = f.ordinal()
    #     r = f.real()

    #     if not math.isnan(r):
    #         print('{}, '.format(r), end='')
    # print()
