"""Sinking point -
A discrete approximation of real numbers with explicit significance tracking.
Implemented really badly in one file.
"""


import typing
import sys
import random
import re

def bitmask(n):
    if n > 0:
        return (1 << n) - 1
    else:
        return (-1) << (-n)


# Binary conversions are relatively simple for numpy's floating point types.
# float16 : w = 5,  p = 11
# float32 : w = 8,  p = 24
# float64 : w = 11, p = 53
# float128: unsupported, not an IEEE 754 128-bit float, possibly 80bit x87?
#           doc says this uses longdouble on the underlying system
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


def xfloat(f):
    if isinstance(f, np.float16):
        w = 5
        pbits = 10
    elif isinstance(f, np.float32):
        w = 8
        pbits = 23
    elif isinstance(f, np.float64):
        w = 11
        pbits = 52
    else:
        raise TypeError('expected np.float{{16,32,64}}, got {}'.format(repr(type(f))))

    emax = (1 << (w - 1)) - 1

    bits = int.from_bytes(f.tobytes(), np_byteorder(type(f)))

    S = bits >> (w + pbits) & bitmask(1)
    E = bits >> (pbits) & bitmask(w)
    C = bits & bitmask(pbits)

    e = E - emax

    if E == 0:
        # subnormal
        return S != 0, -emax - (pbits - C.bit_length()), C
    elif e <= emax:
        # normal
        return S != 0, e, C | (1 << pbits)
    else:
        # nonreal
        raise ValueError('nonfinite value {}'.format(repr(f)))


def mkfloat(s, e, c, ftype=np.float64):
    if ftype == np.float16:
        w = 5
        p = 11
        pbits = 10
        nbytes = 2
    elif ftype == np.float32:
        w = 8
        p = 24
        pbits = 23
        nbytes = 4
    elif ftype == np.float64:
        w = 11
        p = 53
        pbits = 52
        nbytes = 8
    else:
        raise TypeError('expected np.float{{16,32,64}}, got {}'.format(repr(type(f))))

    emax = (1 << (w - 1)) - 1
    emin = 1 - emax

    cbits = c.bit_length()

    if e < emin:
        # subnormal
        lz = (emin - 1) - e
        if lz > pbits or (lz == pbits and cbits > 0):
            raise ValueError('exponent out of range: {}'.format(e))
        elif lz + cbits > pbits:
            raise ValueError('too much precision: given {}, can represent {}'.format(cbits, pbits - lz))
        S = 1 if s else 0
        E = 0
        C = c << (lz - (pbits - cbits))
    elif e <= emax:
        # normal
        if cbits > p:
            raise ValueError('too much precision: given {}, can represent {}'.format(cbits, p))
        elif cbits < p:
            print('Warning: inventing {} low order bits!'.format(p - cbits))
        S = 1 if s else 0
        E = e + emax
        C = (c << (p - cbits)) & bitmask(pbits)
    else:
        # overflow
        raise ValueError('exponent out of range: {}'.format(e))

    return np.frombuffer(
        ((S << (w + pbits)) | (E << pbits) | C).to_bytes(nbytes, np_byteorder(ftype)),
        dtype=ftype, count=1, offset=0,
    )[0]


# debugging

def _nprt(x):
    print(repr(x))
    return mkfloat(*xfloat(x), type(x))

def _check_conv(i, ftype):
    if ftype == np.float16:
        nbytes = 2
    elif ftype == np.float32:
        nbytes = 4
    elif ftype == np.float64:
        nbytes = 8
    else:
        raise TypeError('expected np.float{{16,32,64}}, got {}'.format(repr(type(f))))

    try:
        f = np.frombuffer(i.to_bytes(nbytes, np_byteorder(ftype)), dtype=ftype, count=1, offset=0)[0]
        s, e, c = xfloat(f)
        f2 = mkfloat(s, e, c, ftype)
        if f != f2:
            print(repr(f), repr(f2), s, e, c)
    except ValueError as e:
        if not (np.isinf(f) or np.isnan(f)):
            print(repr(f), repr(f2), s, e, c)
            print('  ' + repr(e))

def _test_np():
    print('watch for output...')
    for i in range(1 << 16):
        _check_conv(i, np.float16)
    for i in range(1 << 16):
        _check_conv(random.randint(0, 1 << 32), np.float32)
    for i in range(1 << 16):
        _check_conv(random.randint(0, 1 << 64), np.float64)
    print('...done')


# gmpy2 helpers
import gmpy2 as gmp
mpfr = gmp.mpfr
mpz = gmp.mpz


def exactctx(prec, emin, emax):
    return gmp.context(
        precision=max(2, prec),
        emin=min(-1, emin),
        emax=max(+1, emax),
        trap_underflow=True,
        trap_overflow=True,
        trap_inexact=True,
        trap_invalid=True,
        trap_erange=True,
        trap_divzero=True,
        trap_expbound=True,
    )


def sigbits(m):
    z = mpz(m)
    trailing_zeros = z.bit_scan1(0)
    if trailing_zeros is None:
        return 0
    else:
        return z.bit_length() - trailing_zeros


def to_shortest_mantissa_exp(r):
    """This destroys info about the precision"""
    m, e = r.as_mantissa_exp()
    trailing_zeros = m.bit_scan1(0)
    if trailing_zeros is None:
        return 0, None
    else:
        return m >> trailing_zeros, e + trailing_zeros


def to_mantissa_exp(r):
    m, e = r.as_mantissa_exp()
    if m == 0:
        return 0, None
    else:
        return m, e


def from_mantissa_exp(m, e):
    if m == 0:
        return mpfr(0, 2)
    mbits = m.bit_length()
    ebits = e.bit_length()
    esig = sigbits(e)
    exp = int(e)
    with exactctx(mbits, min(ebits, mbits, exp), max(ebits, mbits, exp + mbits)):
        rexp = scale = rm = result = None
        try:
            rexp = mpfr(e, max(2, esig))
            scale = mpfr(gmp.exp2(rexp), 2)
            rm = mpfr(m)
            result = gmp.mul(rm, scale)
        except Exception as exc:
            print(exc)
            print(m, e)
            print(mbits, ebits, exp, esig)
            print(result, rm, scale, rexp)
            print(gmp.get_context())
        return result


def withprec(p, op, *args):
    with gmp.context(precision=max(2, p), trap_expbound=True) as gmpctx:
        result = op(*args)
        return result, gmpctx


# more debugging

def _gmprt(x1, x2=None):
    if x2 is None:
        x = mpfr(x1, 53)
    else:
        x = from_mantissa_exp(x1, x2)
    result = from_mantissa_exp(*to_mantissa_exp(x))
    print(repr(x))
    return(x)

def _test_gmp():
    print('watch for output...')
    for i in range(1 << 16):
        m = random.randint(0, 1 << 256) << (random.randint(0, 256) if random.randint(0, 2) else 0)
        e = random.randint(-126, 127) if random.randint(0, 2) else random.randint(-(1 << 32), (1 << 32))
        x1 = from_mantissa_exp(m, e)
        x2 = from_mantissa_exp(*to_mantissa_exp(x1))
        if x1 != x2:
            print(m, e, x1, x2)
        if x1.precision != x2.precision:
            print(x1.precision, x2.precision, x1, x2)
    print('...done')


_DEFAULT_PREC = 53


class Sink:
    _e : int = None # exponent
    _n : int = None # "sticky bit" or lsb
    _p : int = None # precision: e - n
    _c : int = None # significand
    _negative : bool = None # sign bit
    _inexact : bool = None # approximate bit
    _isinf : bool = None # is the value infinite?
    _isnan : bool = None # is this value NaN?


    def _valid(self) -> bool:
        return (
            (self._e >= self._n) and
            (self._p == self._e - self._n) and
            (self._c.bit_length() == self._p) and
            # no support for nonfinite yet
            (not (self._isinf or self._isnan))
        )


    def __init__(self, x=None, e=None, n=None, p=None, c=None,
                 negative=None, inexact=None, isinf=None, isnan=None,
                 max_p=None, min_n=None) -> None:
        """Create a new Sink.
        If an existing Sink is provided, then the fields can be specified individually
        as arguments to the constructor.
        If a new sink is being created, then most fields will be ignored, except n for
        the lsb of 0 values and p for the precision of mpfrs.
        Note that __init__ is currently recursive, to handle some cases of 0 and
        round-on-init with maxp and minn.
        """

        # if given another sink, clone and update
        if isinstance(x, Sink):
            # might have to think about this more carefully...
            self._e = x._e if e is None else e
            self._n = x._n if n is None else n
            self._p = x._p if p is None else p
            self._c = x._c if c is None else c
            self._negative = x._negative if negative is None else negative
            self._inexact = x._inexact if inexact is None else inexact
            self._isinf = x._isinf if isinf is None else isinf
            self._isnan = x._isnan if isnan is None else isnan

        # By default, produce "zero".
        # Note that this throws away the sign of the zero, and substitutes the provided sign...
        # Theoretically we can track one-sided vs two-sided envelopes, but that will take
        # some special logic.
        elif x is None or x == 0:
            if n is None:
                raise ValueError('zero must specify n')
            else:
                self._e = self._n = n
                self._p = self._c = 0
                self._inexact = bool(inexact)
                self._negative = bool(negative)
                self._isinf = self._isnan = False

        # otherwise convert from mpfr
        else:
            # guess precision for
            if p is maxp is None:
                prec = _DEFAULT_PREC
            elif p is None:
                prec = maxp
            else:
                prec = p

            # pi hack
            if isinstance(x, str) and x.strip().lower() == 'pi':
                with gmp.context(precision=prec) as gmpctx:
                    x = gmp.const_pi()
                    inexact = True

            if not isinstance(x, mpfr):
                x = mpfr(x, precision=prec)

            # we reread precision from the mpfr
            m, exp = to_mantissa_exp(x)
            if m == 0:
                # negative is disregarded in this case, only inexact is passed through
                self.__init__(x=0, n=x.precision, inexact=inexact)
            else:
                self._c = abs(int(m))
                self._p = m.bit_length()
                self._n = int(exp) - 1
                self._e = self._n + self._p
                self._inexact = inexact
                self._isinf = self._isnan = False

                if negative is None:
                    self._negative = m < 0
                else:
                    if m < 0:
                        raise ValueError('negative magnitude')
                    self._negative = negative

        if not maxp is minn is None:
            self.__init__(self.widen(min_n=min_n, max_p=max_p))

        assert self._valid()


    def __repr__(self):
        try:
            mpfr_val = self.as_mpfr()
        except Exception as exc:
            mpfr_val = exc
        try:
            f64_val = self.as_np(np.float64)
        except Exception as exc:
            f64_val = exc
        try:
            f32_val = self.as_np(np.float32)
        except Exception as exc:
            f32_val = exc
        try:
            f16_val = self.as_np(np.float16)
        except Exception as exc:
            f16_val = exc
        return ('Sinking point number:\n  e={}\n  n={}\n  p={}\n  c={}\n  negative={}\n  inexact={}\n  isinf={}\n  isnan={}\n  valid? {}'
                .format(self._e, self._n, self._p, self._c, self._negative, self._inexact, self._isinf, self._isnan, self._valid()) +
                '\n  as mpfr: {}\n  as np.float64: {}\n  as np.float32: {}\n  as np.float16: {}'
                .format(repr(mpfr_val), repr(f64_val), repr(f32_val), repr(f16_val)))


    def __str__(self):
        """yah"""
        if self._c == 0:
            sgn = '-' if self._negative else ''
            if self._inexact:
                return '{}0~@{:d}'.format(sgn, self._n)
            else:
                return '{}0'.format(sgn)
        else:
            rep = re.search(r"'(.*)'", repr(self.as_mpfr())).group(1).split('e')
            s = rep[0]
            sexp = ''
            if len(rep) > 1:
                sexp = 'e' + 'e'.join(rep[1:])
            return '{}{}{}'.format(s, '~' if self._inexact else '', sexp)
            # return '{}{}'.format(rep, '~@{:d}'.format(self._n) if self._inexact else '')


    # properties


    def is_exactly_zero(self) -> bool:
        """Really there are multiple kinds of 0:
          - 'Exactly' 0, as written
          - 0 or infinitely close to 0, from either side: lim(n) as n -> 0
          - finitely close to 0, from either side: lim(n) as n -> small
          - finitely close to zero from some side, side unknown
        """
        return self._c == 0 and (not self._inexact)


    # core envelope operations


    def collapse(self):
        """Collapse the interval down to its average.
        Currently just clears the inexact flag, could do something more clever.
        """
        return Sink(self, inexact=False)


    def explode(self):
        """Explodes a point back to an enclosing interval.
        Currently just sets the inexact flag, could do something more clever.
        """
        return Sink(self, inexact=True)


    def away(self, const_p = False) -> Sink:
        """The sink with the next greatest magnitude at this precision, away from 0.
        Preserves sign and exactness.
        Theoretically meaningless for non-sided zero.
        """
        next_e = self._e
        next_c = self._c + 1
        next_n = self._n
        next_p = self._p

        if next_c.bit_length() > self._c.bit_length():
            # adjust e if we carried
            next_e += 1
            if const_p and next_c > 1:
                # normalize precision, if we want to keep it constant
                # only possible if we didn't start from 0
                next_c >>= 1
                next_n += 1
            else:
                next_p += 1

        return Sink(self, e=next_e, n=next_n, p=next_p, c=next_c)


    def toward(self, const_p = False) -> Sink:
        """The sink with the next smallest magnitude at this precision, toward 0.
        Preserves sign and exactness.
        Meaningless for any zero.
        """
        prev_e = self._e
        prev_c = self._c - 1
        prev_n = self._n
        prev_p = self._p

        if prev_c < 0:
            raise ValueError('toward: {} is already 0'.format(repr(self)))

        if prev_c.bit_length() < self._c.bit_length():
            # adjust e if we borrowed
            prev_e -= 1
            if const_p and prev_c > 0:
                # normalize precision, if we want to keep it constant
                # only possible if we didn't actually reach 0
                prev_c <<= 1
                prev_n -= 1
            else:
                prev_p -= 1

        return Sink(self, e=prev_e, n=prev_n, p=prev_p, c=prev_c)


    def above(self, const_p = False) -> Sink:
        """The sink with the next largest value, toward positive infinity.
        Special cases for 0 (especially if sided).
        """
        if self._c == 0:
            # theoretically this should handle the sided case, right now it doesn't
            return Sink(self, e=self._n+1, p=1, c=1, negative=False)
        elif self._negative:
            return self.toward(const_p=const_p)
        else:
            return self.away(const_p=const_p)


    def below(self, const_p = False) -> Sink:
        """The sink with the next smallest value, toward negative infinity.
        Special cases for 0 (especially if sided).
        """
        if self._c == 0:
            # theoretically this should handle the sided case, right now it doesn't
            return Sink(self, e=self._n+1, p=1, c=1, negative=True)
        elif self._negative:
            return self.away(const_p=const_p)
        else:
            return self.toward(const_p=const_p)


    def bounds(self) -> Sink:
        """Upper and lower bounds on the value of this number.
        Note that this could do something clever with half intervals, but it doesn't.
        """
        if self._inexact:
            return self.above().collapse(), self.below().collapse()
        else:
            return Sink(self), Sink(self)


    def split(self, n=None, rm=0) -> typing.Tuple[Sink, Sink]:
        """Split a number into an exact part and an uncertainty bound.
        If we produce split(A, n) -> A', E, then we know:
          - A' is exact
          - E is zero
          - lsb(A') == lsb(E) == max(n, lsb(A)) if A is inexact
          - lsb(A') == lsb(E) == n if A is exact
        """
        if n is None:
            n = self._n
        offset = n - self._n

        if offset <= 0:
            if offset == 0 or self._inexact:
                return (
                    Sink(self, inexact=False),
                    Sink(0, n=self._n, negative=self._negative, inexact=self._inexact),
                )
            else:
                extended_c = self._c << -offset
                extended_p = extended_c.bit_length()
                extended_e = n if extended_c == 0 else self._e
                return (
                    Sink(self, e=extended_e, n=n, p=extended_p, c=extended_c),
                    Sink(0, n=n, negative=self._negative, inexact=self._inexact),
                )
        else:
            lost_bits = self._c & bitmask(offset)
            left_bits = self._c >> offset
            low_bits = lost_bits & bitmask(offset - 1)
            half_bit = lost_bits >> (offset - 1)

            e = max(self._e, n)
            inexact = self._inexact or lost_bits != 0

            rounded = Sink(self, e=e, n=n, p=e-n, c=left_bits, inexact=False)
            # in all cases we copy the sign onto epsilon... is that right?
            epsilon = Sink(0, n=n, negative=self._negative, inexact=inexact)

            if half_bit == 1:
                # Note that if we're rounding an inexact number, then the new tight 1-ulp envelope
                # of the result will not contain the entire envelope of the input.
                if low_bits == 0:
                    # Exactly half way between, regardless of exactness.
                    # Use rounding mode to decide.
                    if rm == 0:
                        # round to even if rm is zero
                        if left_bits & bitmask(1) == 1:
                            return rounded.away(const_p=False), epsilon
                        else:
                            return rounded, epsilon
                    elif rm > 0:
                        # round away from zero if rm is positive
                        return rounded.away(const_p=False), epsilon
                    else:
                        # else, round toward zero if rm is negative
                        return rounded, epsilon
                else:
                    return rounded.away(const_p=False), epsilon
            else:
                return rounded, epsilon


    def widen(self, min_n = None, max_p = None) -> Sink:
        """Round this number, using split, so that n is >= min_n and p <= max_p.
        By default, preserve n and p, returning this number unchanged.
        """
        if min_n is None:
            n = self._n
        else:
            n = min_n

        if max_p is not None:
            n = max(n, self._e - maxp)

        rounded, epsilon = self.split(n)

        if max_p is not None and rounded._p > max_p:
            # If we rounded up and carried, we might have increased p by one.
            # Split again to compensate; this should produce an epsilon of zero.
            rounded, epsilon_correction = rounded.split(n + 1)
            if not epsilon_correction.is_exactly_zero():
                epsilon = epsilon_correction
                raise ValueError('widen: unreachable')

        rounded._inexact = epsilon._inexact
        return rounded


    def narrow(self, n=None, p=None) -> Sink:
        """Force this number into a representation with either n or p.
        By default, preserve n and p, returning this number unchanged.
        Note that this may produce a smaller envelope that does not contain
        the input value.
        """
        if n is p is None:
            return Sink(self)
        elif n is None:
            if self._c == 0:
                # specifying precision is meaningless for zero
                n = self._n
            else:
                n = self._e - p
        elif p is None:
            # use n as provided
            pass
        else:
            raise ValueError('narrow: can only specify one of n or p, got n={}, p={}'
                             .format(repr(n), repr(p)))

        rounded, epsilon = self.split(n)

        # There are two possibilities:
        # Either we are trying to narrow the envelope, i.e. increase precision,
        # and this split was a no-op;
        # Or we are actually trying to widen the envelope, i.e. decrease precision,
        # and this split may have rounded up, giving us more precision than we want.

        if rounded._n > n:
            # split was unable to provide a small enough n, so we have to force one
            extended_c = rounded._c << (rounded._n - n)
            extended_p = extended_c.bit_length()
            extended_e = n if extended_c == 0 else rounded._e
            rounded = Sink(rounded, e=extended_e, n=n, p=extended_p, c=extended_c)
        elif p is not None and rounded._p > p:
            # as for widening, round again to compensate
            rounded, epsilon_correction = rounded.split(n+1)
            if not epsilon_correction.is_exactly_zero():
                epsilon = epsilon_correction
                raise ValueError('narrow: unreachable')

        rounded._inexact = epsilon._inexact
        return rounded


    def as_mpfr(self):
        return from_mantissa_exp(self._c * (-1 if self._negative else 1), self._n + 1)


    def as_np(self, ftype=np.float64):
        # TODO: breaks for 0
        return mkfloat(self._negative, self._e, self._c, ftype=ftype)


    # arith, for now, only optimistic precision bounds


    def __neg__(self):
        # oh gawd
        sunk = Sink(None)
        sunk._e = self._e
        sunk._n = self._n
        sunk._p = self._p
        sunk._c = self._c
        sunk._negative = not self._negative
        sunk._inexact = self._inexact
        sunk._isinf = self._isinf
        sunk._isnan = self._isnan
        return sunk


    def __add__(self, arg):
        # slow and scary
        prec = (max(self._e, arg._e) - min(self._n, arg._n)) + 1
        # could help limit with this?
        if (not self._inexact) and (not arg._inexact):
            n = None
        elif self._inexact and (not arg._inexact):
            n = self._n
        elif arg._inexact and (not self._inexact):
            n = arg._n
        else:
            n = max(self._n, arg._n)
        result_f, ctx = withprec(prec, gmp.add, self.as_mpfr(), arg.as_mpfr())
        result = Sink(result_f, p=prec, negative=(result_f < 0),
                      inexact=(ctx.inexact or self._inexact or arg._inexact))
        # mandatory rounding even for optimists:
        return result.trunc(n)


    def __sub__(self, arg):
        return self + (-arg)


def adjacent_mpfrs(x):
    # so this is completely broken
    yield x


def inbounds(lower, upper):
    """pls to give mpfrs... nope we need to use sinks herp derp"""
    if upper < lower:
        raise ValueError('invalid bounding range [{}, {}]'.format(upper))
    elif lower == upper:
        # TODO: breaks for -0
        return Sink(lower, p=lower.precision, negative=lower<0, inexact=False)
    else:
        # sterbenz applies here
        prec = max(lower.precision, upper.precision)
        difference = withprec(prec, gmp.sub, upper, lower)
        # retain another bit
        prec += 1
        half_difference = withprec(prec, gmp.div, difference, 2)
        mid = withprec(prec, gmp.add, lower, half_difference)
        # TODO: linear scan
        while prec > 2:
            pass


# halp

def ___ctx():
    gmp.set_context(gmp.context())


"""TODO:
  Sound half-intervals:
  - sided zero
  - 1 ulp vs. 2 ulp
  Arithmetic emulation:
  - 754
  - posit
  Bit savings:
  - but we would have to count on GMP's rounding behavior
"""


def addsub_mpfr(a, b):
    """(a + b) - a"""
    ___ctx()
    A = mpfr(a)
    B = mpfr(b)
    result = (A + B) - A
    return result


def addsub_exact(a, b):
    """(a + b) - a"""
    A = Sink(a, inexact=False)
    B = Sink(b, inexact=False)
    result = (A + B) - A
    return str(result)


def addsub_sink(a, a_inexact, b, b_inexact, maxp=None):
    """(a + b) - a"""
    A = Sink(a, inexact=a_inexact)
    B = Sink(b, inexact=b_inexact)
    A_B = (A + B).trunc(maxp=maxp)
    result = (A_B - A).trunc(maxp=maxp)
    return str(result)


def addsub_limited(a, b):
    A = Sink(a, inexact=False)
    B = Sink(b, inexact=False)
    A_B = (A + B).trunc(maxp=53)
    result = (A_B - A).trunc(maxp=53)
    return str(result)


___ctx()
pie = gmp.const_pi()
