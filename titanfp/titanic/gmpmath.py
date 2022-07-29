"""Common arithmetic operations (+-*/ sqrt log exp etc.)
implemented with GMP as a backend, but conveniently extended to Sinking Point.
"""


import gmpy2 as gmp
import re

from . import utils

from .integral import bitmask
from . import conversion
from . import ops
from . import digital
from .sinking import Sink


def mpfr(x, prec):
    with gmp.context(
            # one extra bit, so that we can round from RTZ to RNE
            precision=prec + 1,
            emin=gmp.get_emin_min(),
            emax=gmp.get_emax_max(),
            trap_underflow=True,
            trap_overflow=True,
            trap_inexact=False,
            trap_invalid=True,
            trap_erange=True,
            trap_divzero=True,
            # use RTZ for easy multiple rounding later
            round=gmp.RoundToZero,
    ):
        return gmp.mpfr(x)


def digital_to_mpfr(x):
    if x.isnan:
        return gmp.nan()
    elif x.isinf:
        if x.negative:
            return -gmp.inf()
        else:
            return gmp.inf()

    c = x.c
    exp = x.exp

    cbits = c.bit_length()
    ebits = exp.bit_length()

    # Apparently a multiplication between a small precision 0 and a huge
    # scale can raise a Type error indicating that gmp.mul() requires two
    # mpfr arguments - we can avoid that case entirely by special-casing
    # away the multiplication.
    if cbits == 0:
        with gmp.context(
            precision=2,
            emin=-1,
            emax=1,
            trap_underflow=True,
            trap_overflow=True,
            trap_inexact=True,
            trap_invalid=True,
            trap_erange=True,
            trap_divzero=True,
        ):
            if x.negative:
                return -gmp.zero()
            else:
                return gmp.zero()

    else:
        with gmp.context(
                precision=max(2, ebits),
                emin=min(-1, exp),
                emax=max(1, ebits, exp + 1),
                trap_underflow=True,
                trap_overflow=True,
                trap_inexact=True,
                trap_invalid=True,
                trap_erange=True,
                trap_divzero=True,
        ):
            scale = gmp.exp2(exp)

        with gmp.context(
                precision=max(2, cbits),
                emin=min(-1, exp),
                emax=max(1, cbits, exp + cbits),
                trap_underflow=True,
                trap_overflow=True,
                trap_inexact=True,
                trap_invalid=True,
                trap_erange=True,
                trap_divzero=True,
        ):
            significand = gmp.mpfr(c)
            if x.negative:
                return -gmp.mul(significand, scale)
            else:
                return gmp.mul(significand, scale)


def mpfr_to_digital(x):
    rounded = x.rc != 0

    if gmp.is_nan(x):
        return digital.Digital(
            isnan=True,
            inexact=rounded,
            rounded=rounded,
            rc=0, # TODO remove me
        )

    negative = gmp.is_signed(x)

    # Convert the result code. For MPFRs, 1 indicates that the approximate MPFR
    # is larger than the ideal, infinite-precision result (i.e. we rounded up)
    # and -1 indicates that the MPFR is less than the infinite-precision result.
    # We need to convert this to a different code used by titanic: the result
    # code is a tiny additional factor that we would have to add to the magnitude to
    # get the right answer, so if we rounded away from zero, it's -1, and if we rounded
    # towards zero, it's 1.

    if negative:
        rc = x.rc
    else:
        rc = -x.rc

    if gmp.is_infinite(x):
        return digital.Digital(
            negative=negative,
            isinf=True,
            inexact=rounded,
            rounded=rounded,
            rc=rc, # TODO remove me
        )

    m, exp = x.as_mantissa_exp()
    c = int(abs(m))
    exp = int(exp)

    if c == 0:
        assert rc != -1, ('unreachable: MPFR rounded the wrong way toward zero? got {}, rc={}'
                         .format(repr(x), repr(x.rc)))

    # work out the new envelope parameters
    if rounded:
        # There's no way to figure out what rounding mode was used to produce this MPFR,
        # so we assume it was rounded towards zero.
        # Titanic always rounds its MPFR calculations in this way, and uses one extra
        # bit of precision so it reproduce rounding to nearest, and also detect
        # results that are exactly halfway between representable numbers.
        # TODO: is there a weird case here for c == 1???
        if rc > 0 and (c & 1 == 0) and (c.bit_length() > (c-1).bit_length()):
            # rounded up to a power of two; shrink envelope
            interval_size = -1
        else:
            # assume rounding to zero, not to nearest; envelope size is a whole ulp
            interval_size = 0
        interval_down = (rc < 0)
        interval_closed = False
    else:
        interval_size = 0
        interval_down = False
        interval_closed = False

    return digital.Digital(
        negative=negative,
        c=c,
        exp=exp,
        inexact=rounded,
        rounded=rounded,
        rc=rc, # TODO remove me
        interval_size=interval_size,
        interval_down=interval_down,
        interval_closed=interval_closed,
    )


def _fdim(x1, x2):
    raise ValueError('fdim: emulated')
def _fmax(x1, x2):
    raise ValueError('fmax: emulated')
def _fmin(x1, x2):
    raise ValueError('fmin: emulated')

gmp_ops = [
    gmp.add,
    gmp.sub,
    gmp.mul,
    gmp.div,
    lambda x: -x,
    gmp.sqrt,
    gmp.fma,
    gmp.copy_sign,
    lambda x: abs(x),
    _fdim,
    _fmax,
    _fmin,
    gmp.fmod,
    gmp.remainder,
    gmp.ceil,
    gmp.floor,
    gmp.rint,
    gmp.round_away,
    gmp.trunc,
    gmp.acos,
    gmp.acosh,
    gmp.asin,
    gmp.asinh,
    gmp.atan,
    gmp.atan2,
    gmp.atanh,
    gmp.cos,
    gmp.cosh,
    gmp.sin,
    gmp.sinh,
    gmp.tan,
    gmp.tanh,
    gmp.exp,
    gmp.exp2,
    gmp.expm1,
    gmp.log,
    gmp.log10,
    gmp.log1p,
    gmp.log2,
    gmp.cbrt,
    gmp.hypot,
    lambda x1, x2: x1 ** x2,
    gmp.erf,
    gmp.erfc,
    lambda x: gmp.lgamma(x)[0],
    gmp.gamma,
]


def compute(opcode, *args, prec=53):
    """Compute op(*args), with up to prec bits of precision.
    op is specified via opcode, and arguments are universal digital numbers.
    Arguments are treated as exact: the inexactness and result code of the result
    only reflect what happened during this single operation.
    Result is truncated towards 0, but will have inexactness and result code set
    for further rounding, and it is computed with one extra bit of precision.
    NOTE: this function does not trap on invalid operations, so it will give the gmp/mpfr answer
    for special cases like sqrt(-1), arcsin(3), and so on.
    """
    op = gmp_ops[opcode]
    inputs = [digital_to_mpfr(arg) for arg in args]
    # gmpy2 really doesn't like it when you pass nan as an argument
    for f in inputs:
        if gmp.is_nan(f):
            return mpfr_to_digital(f)
    with gmp.context(
            # one extra bit, so that we can round from RTZ to RNE
            precision=prec + 1,
            emin=gmp.get_emin_min(),
            emax=gmp.get_emax_max(),
            subnormalize=False,
            # in theory, we'd like to know about these...
            trap_underflow=True,
            trap_overflow=True,
            # inexact and invalid operations should not be a problem
            trap_inexact=False,
            trap_invalid=False,
            trap_erange=False,
            trap_divzero=False,
            # We'd really like to know about this as well, but it causes i.e.
            #   mul(-25, inf) -> raise TypeError("mul() requires 'mpfr','mpfr' arguments")
            # I don't know if that behavior is more hilarious or annoying.
            # This context property was removed in gmpy2 2.1
            #trap_expbound=False,
            # use RTZ for easy multiple rounding later
            round=gmp.RoundToZero,
    ) as gmpctx:
        result = op(*inputs)

    return mpfr_to_digital(result)


constant_exprs = {
    'E' : lambda : gmp.exp(1),
    'LOG2E' : lambda: gmp.log2(gmp.exp(1)), # TODO: may be inaccurate
    'LOG10E' : lambda: gmp.log10(gmp.exp(1)), # TODO: may be inaccurate
    'LN2' : gmp.const_log2,
    'LN10' : lambda: gmp.log(10),
    'PI' : gmp.const_pi,
    'PI_2' : lambda: gmp.const_pi() / 2, # division by 2 is exact
    'PI_4' : lambda: gmp.const_pi() / 4, # division by 4 is exact
    'M_1_PI' : lambda: 1 / gmp.const_pi(), # TODO: may be inaccurate
    'M_2_PI' : lambda: 2 / gmp.const_pi(), # TODO: may be inaccurate
    'M_2_SQRTPI' : lambda: 2 / gmp.sqrt(gmp.const_pi()), # TODO: may be inaccurate
    'SQRT2': lambda: gmp.sqrt(2),
    'SQRT1_2': lambda: gmp.sqrt(gmp.div(gmp.mpfr(1), gmp.mpfr(2))),
    'INFINITY': gmp.inf,
    'NAN': gmp.nan,
}

def compute_constant(name, prec=53):
    with gmp.context(
            # TODO: a few extra bits, so that it hopefully works out
            precision=prec + 5,
            emin=gmp.get_emin_min(),
            emax=gmp.get_emax_max(),
            subnormalize=False,
            # in theory, we'd like to know about these...
            trap_underflow=True,
            trap_overflow=True,
            # inexact and invalid operations should not be a problem
            trap_inexact=False,
            trap_invalid=False,
            trap_erange=False,
            trap_divzero=False,
            # We'd really like to know about this as well, but it causes i.e.
            #   mul(-25, inf) -> raise TypeError("mul() requires 'mpfr','mpfr' arguments")
            # I don't know if that behavior is more hilarious or annoying.
            # This context property was removed in gmpy2 2.1
            #trap_expbound=False,
            # use RTZ for easy multiple rounding later
            round=gmp.RoundToZero,
    ) as gmpctx:
        try:
            result = constant_exprs[name]()
        except KeyError as e:
            raise ValueError('unknown constant {}'.format(repr(e.args[0])))

    return mpfr_to_digital(result)


def compute_digits(m, e, b, prec=53):
    """Compute m * b**e, with precision equal to prec. e and b must be integers, and
    b must be at least 2.
    """
    if (not isinstance(e, int)) or (not isinstance(b, int)) or (b < 2):
        raise ValueError('compute_digits: must have integer e, b, and b >= 2, got e={}, b={}'
                         .format(repr(e), repr(b)))

    with gmp.context(
            precision=max(e.bit_length(), b.bit_length()),
            emin=gmp.get_emin_min(),
            emax=gmp.get_emax_max(),
            trap_underflow=True,
            trap_overflow=True,
            trap_inexact=True,
            trap_invalid=True,
            trap_erange=True,
            trap_divzero=True,
            round=gmp.RoundToZero,
    ) as gmpctx:
        mpfr_e = gmp.mpfr(e)
        mpfr_b = gmp.mpfr(b)

    with gmp.context(
            # this seems like it's enough extra bits, but I don't have a proof
            precision=prec + 3,
            emin=gmp.get_emin_min(),
            emax=gmp.get_emax_max(),
            trap_underflow=True,
            trap_overflow=True,
            trap_inexact=False,
            trap_invalid=True,
            trap_erange=True,
            trap_divzero=True,
            # use RTZ for easy multiple rounding later
            round=gmp.RoundToZero,
    ) as gmpctx:
        mpfr_m = gmp.mpfr(m)
        scale = mpfr_b ** mpfr_e
        result = mpfr_m * scale

    return mpfr_to_digital(result)


def ieee_fbound(w, p):
    """Compute the boundary where IEEE 754 floating-point values
    will be rounded away to infinity for a given w and p.
    """
    emax = (1 << (w - 1)) - 1

    with gmp.context(
            precision=p + 1,
            emin=gmp.get_emin_min(),
            emax=gmp.get_emax_max(),
            trap_underflow=True,
            trap_overflow=True,
            trap_inexact=True,
            trap_invalid=True,
            trap_erange=True,
            trap_divzero=True,
    ):
        fbound_scale = gmp.mpfr(2) - gmp.exp2(-p)
        fbound = gmp.exp2(emax) * fbound_scale

    return mpfr_to_digital(fbound)

def ieee_fmax(w, p):
    """Compute the the largest finite IEEE 754 floating-point value
    for a given w and p.
    """
    emax = (1 << (w - 1)) - 1

    with gmp.context(
            precision=p + 1,
            emin=gmp.get_emin_min(),
            emax=gmp.get_emax_max(),
            trap_underflow=True,
            trap_overflow=True,
            trap_inexact=True,
            trap_invalid=True,
            trap_erange=True,
            trap_divzero=True,
    ):
        fmax_scale = gmp.mpfr(2) - gmp.exp2(1 - p)
        fmax = gmp.exp2(emax) * fmax_scale

    return mpfr_to_digital(fmax)


_mpz_2 = gmp.mpz(2)
_mpz_5 = gmp.mpz(5)
_mpz_10 = gmp.mpz(10)
def decimal_expansion(x):
    if x.is_zero():
        return 0
    else:
        c = x.c
        c2, twos = gmp.remove(c, 2)
        exp2 = x.exp + twos
        # at this point, x == c2 * (2**exp2)

        c5, fives = gmp.remove(c2, 5)
        # x == c5 * (2**exp2) * (5**fives)
        # fives is positive, but exp2 might be negative

        if exp2 >= 0:
            tens = min(exp2, fives)
            exp2 -= tens
            fives -= tens
            # x == c5 * (2**exp2) * (5**fives) * (10**tens)
            return int(c5 * (_mpz_2 ** exp2) * (_mpz_5 ** fives)), tens

        else:
            # x == (c5 * (5**fives) * (5**-exp2)) / ((2**-exp2) * (5**-exp2))
            return int(c5 * (_mpz_5 ** (fives - exp2))), exp2


def dec_to_str(c_tens):
    c, tens = c_tens
    if tens >= 0:
        return str(c) + ('0' * tens) + '.'
    else:
        s = str(c)
        if len(s) <= -tens:
            return '.' + ('0' * -(len(s) + tens)) + s
        else:
            return s[:tens] + '.' + s[tens:]

def str_to_dec(s):
    point = s.find('.')
    if point == -1: # no decimal point found
        si = s
        tens = 0
    else:
        si = s[:point] + s[point+1:]
        tens = point - len(si)

    if si.endswith('0'):
        s10 = si.rstrip('0')
        tens += (len(si) - len(s10))
        return int(s10), tens
    else:
        return int(si), tens

def round_dec(d, p):
    """correctly round a decimal expansion d = c, tens to p digits"""
    c, tens = d
    ndig = len(str(c))

    if ndig <= p:
        offset = p - ndig
        return [(c * (10 ** offset), tens - offset)]
    else:
        offset = ndig - p
        scale = _mpz_10 ** offset
        tens_offset = tens + offset

        floor, rem = gmp.f_divmod(c, scale)
        halfrem = rem - gmp.f_div(scale, 2)

        if halfrem < 0:
            return [(int(floor), tens_offset)]
        elif halfrem == 0:
            # half way - try both
            return [(int(floor), tens_offset), (int(floor+1), tens_offset)]
        else:
            return [(int(floor+1), tens_offset)]

def round_dec_liberally(d, p):
    c, tens = d
    ndig = len(str(c))

    if ndig <= p:
        offset = p - ndig
        tens_offset = tens - offset
        new_c = int(gmp.mpz(c) * (_mpz_10 ** offset))
    else:
        offset = ndig - p
        scale = _mpz_10 ** offset
        tens_offset = tens + offset
        floor, rem = gmp.f_divmod(c, scale)
        new_c = int(floor)
    return [(new_c - 1, tens_offset), (new_c, tens_offset), (new_c + 1, tens_offset)]





# helpful constants we don't need to constantly redefine
_mpz_2 = gmp.mpz(2)
_mpz_5 = gmp.mpz(5)
_mpz_10 = gmp.mpz(10)
#                       1         2           3         4               5
_dec_re = re.compile(r'([-+]?)(?:([0-9]+)\.?|([0-9]*)\.([0-9]+))(?:[eE]([+-]?[0-9]+))?')

class Dec(object):
    """Exact decimal representation of a number, with support for rounding.
    Intended for reading or printing decimal strings.
    """

    # raw parameters
    _negative = False
    _c = 0
    _exp = 0

    # cached parameters: only access through properties
    _q = None
    _e = None
    _s = None
    _digits = None
    _string = None
    _estring = None

    @property
    def negative(self):
        """The sign: is this number decimal than zero?"""
        return self._negative

    @property
    def c(self):
        """The unsigned significand (or numerator) of the decimal."""
        return self._c

    @property
    def exp(self):
        """The (decimal) exponent of the decimal."""
        return self._exp

    @property
    def tens(self):
        """Alias for the exponent of the decimal."""
        return self._exp

    @property
    def p(self):
        """The numerator of the decimal fraction."""
        return self._c

    @property
    def q(self):
        """The denominator of the decimal fraction."""
        if self._q is None:
            self._q = int(_mpz_10 ** self._exp)
        return self._q

    @property
    def s(self):
        """The string representation of the numerator of this decimal."""
        if self._s is None:
            self._s = str(gmp.mpz(self._c))
            if self._c == 0:
                self._digits = 0
            else:
                self._digits = len(self._s)
        return self._s

    @property
    def digits(self):
        """The number of digits (in the numerator) of this decimal."""
        if self._digits is None:
            self._s = str(gmp.mpz(self._c))
            if self._c == 0:
                self._digits = 0
            else:
                self._digits = len(self._s)
        return self._digits

    @property
    def e(self):
        """The scientific notation-style exponent of this decimal (as in 100 = 1.00e2)."""
        if self._e is None:
            if self.digits == 0:
                self._e = self._exp
            else:
                self._e = self._exp + self.digits - 1
        return self._e

    @property
    def string(self):
        """The string representation of this decimal, in decimal (%f) notation."""
        if self._string is None:
            if self._c == 0:
                if self._exp >= 0:
                    self._string =  ('0' * (self._exp + 1)) + '.'
                else:
                    self._string = '.' + ('0' * -self._exp)
            elif self._exp >= 0:
                self._string = self.s + ('0' * self._exp) + '.'
            else:
                s = self.s
                digits = self.digits
                if digits <= -self._exp:
                    self._string = '.' + ('0' * -(digits + self._exp)) + s
                else:
                    self._string = s[:self._exp] + '.' + s[self._exp:]

            if self._negative:
                self._string = '-' + self._string

        return self._string

    @property
    def estring(self):
        """The string representation of this decimal, in scientific (%e) notation."""
        if self._estring is None:
            if self.e > 0:
                expstr = 'e+' + str(self.e)
            else:
                expstr = 'e' + str(self.e)

            s = self.s
            if self._negative:
                self._estring = '-' + s[:1] + '.' + s[1:] + expstr
            else:
                self._estring = s[:1] + '.' + s[1:] + expstr

        return self._estring

    def __init__(self, x=None, tens=None, negative=None):
        """Create a new decimal from a digital number or string (one argument)
        or from a particular significand and decimal exponent (two arguments, in that order).
        """
        if tens is None and negative is None:
            if isinstance(x, str):
                m = _dec_re.fullmatch(x)
                if m is None:
                    raise ValueError('invalid decimal literal {}'.format(repr(x)))

                self._negative = (m.group(1) == '-')

                if m.group(2) is not None:
                    self._c = int(m.group(2))
                    if self._c == 0:
                        exp = len(m.group(2)) - 1
                    else:
                        exp = 0
                else:
                    self._c = int(m.group(3) + m.group(4))
                    exp = -len(m.group(4))

                if m.group(5) is not None:
                    self._exp = exp + int(m.group(5))
                else:
                    self._exp = exp

            elif x is None:
                self._negative = False
                self._c = 0
                self._exp = 0

            else:
                self._negative = x.negative
                # this code also simplifies
                if x.is_zero():
                    self._c = 0
                    self._exp = 0
                else: # x is a nonzero digital
                    c = x.c
                    c2, twos = gmp.remove(c, 2)
                    exp2 = x.exp + twos
                    # at this point, x == c2 * (2**exp2)

                    c5, fives = gmp.remove(c2, 5)
                    # x == c5 * (2**exp2) * (5**fives)
                    # fives is positive, but exp2 might be negative

                    if exp2 >= 0:
                        tens = min(exp2, fives)
                        exp2 -= tens
                        fives -= tens
                        # x == c5 * (2**exp2) * (5**fives) * (10**tens)
                        self._c = int(c5 * (_mpz_2 ** exp2) * (_mpz_5 ** fives))
                        self._exp = tens
                    else:
                        # x == (c5 * (5**fives) * (5**-exp2)) / ((2**-exp2) * (5**-exp2))
                        self._c = int(c5 * (_mpz_5 ** (fives - exp2)))
                        self._exp = exp2

        elif x is not None and tens is not None:
            # if negative is not given, then the decimal will be positive
            self._negative = bool(negative)
            self._c = int(x)
            self._exp = int(tens)

        else:
            raise ValueError('invalid arguments for Dec: x={}, tens={}, negative={}'
                             .format(repr(x), repr(tens), repr(negative)))

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self._c) + ', ' + repr(self._exp) + ', negative=' + repr(self._negative) + ')'

    def __str__(self):
        return self.string

    def simplify(self):
        """Find the largest exponent that can represent this number exactly,
        removing all trailing zeros.
        """
        if self.c == 0:
            c10, tens = 0, -self.exp
        else:
            c10, tens = gmp.remove(self.c, 10)

        if tens == 0:
            return self
        else:
            return Dec(c10, self.exp + tens, negative=self.negative)

    def round(self, p, direction=None):
        """Round to exactly p digits, according to direction:
           None: nearest
             +1: round up
              0: truncate
        """
        digits = self.digits

        if digits <= p:
            offset = p - digits
            return Dec(self.c * (_mpz_10 ** offset), self.exp - offset, negative=self.negative)
        else:
            offset = digits - p
            scale = _mpz_10 ** offset
            new_exp = self.exp + offset

            floor, rem = gmp.f_divmod(self.c, scale)
            # halfrem = rem - gmp.f_div(scale, 2)

            if direction == 0 or rem == 0:
                return Dec(floor, new_exp, negative=self.negative)

            elif direction == 1:
                new_c = floor + 1
                attempt = Dec(new_c, new_exp, negative=self.negative)
                if attempt.digits > p:
                    attempt = Dec(gmp.f_div(new_c, 10), new_exp - 1, negative=self.negative)
                return attempt

            elif direction == None:
                halfrem = rem - gmp.f_div(scale, 2)
                if halfrem < 0:
                    return Dec(floor, new_exp, negative=self.negative)
                elif halfrem == 0:
                    # half way - round to even
                    if gmp.is_even(floor):
                        return Dec(floor, new_exp, negative=self.negative)
                    else:
                        new_c = floor + 1
                        attempt = Dec(new_c, new_exp, negative=self.negative)
                else: # halfrem > 0
                    new_c = floor + 1
                    attempt = Dec(new_c, new_exp, negative=self.negative)

                # at this point, we either returned or set new_c and attempt
                if attempt.digits > p:
                    attempt = Dec(gmp.f_div(new_c, 10), new_exp - 1, negative=self.negative)
                return attempt

            else:
                raise ValueError('invalid rounding direction {} for {}'
                                 .format(repr(direction), repr(self)))

    def scale(self, exp):
        """Scale so that the exponent is exactly exp.
        Will fail if the provided exponent is larger than the current one - always scale to the minimum!
        """
        if exp < self.exp:
            return Dec(self.c * (_mpz_10 ** (self.exp - exp)), exp, negative=self.negative)
        elif exp == self.exp:
            return self
        else:
            raise ValueError('cannot scale {} to exp {}: target exponent too large'
                             .format(repr(self), repr(exp)))

    def normalize_to(self, other):
        """Find the largest common exponent that can be used to represent both self and other;
        return two new decimals scaled to this exponent.
        """
        d1 = self.simplify()
        d2 = other.simplify()
        exp = min(d1.exp, d2.exp)
        return d1.scale(exp), d2.scale(exp)


#                             1      2          3      4         5      6          7
_dec_range_re = re.compile(r'([-+]?)([.0-9]*)\[([+-]?)([.0-9]+)-([+-]?)([.0-9]+)\]((?:[eE][+-]?[0-9]+)?)')
def str_to_dec_range(s):
    m = _dec_range_re.fullmatch(s)
    if m is None:
        raise ValueError('invalid decimal range literal {}'.format(repr(s)))

    if (m.group(1) or m.group(2)) and (m.group(3) or m.group(5)):
        raise ValueError('too many signs in decimal range literal {}'.format(repr(s)))

    if '.' in m.group(4) or '.' in m.group(6):
        if '.' in m.group(2):
            raise ValueError('too many dots in decimal range literal {}'.format(repr(s)))
        if m.group(4) == '.' or m.group(6) == '.':
            raise ValueError('empty range in decimal range literal {}'.format(repr(s)))

    s1 = ''.join(m.group(1,2,3,4,7))
    s2 = ''.join(m.group(1,2,5,6,7))

    return Dec(s1), Dec(s2)

def dec_range_to_str(d1, d2, scientific=False):
    # normalizing cuts off any trailing zeros
    d1, d2 = d1.normalize_to(d2)
    exp = d1.exp

    if d1.c == d2.c and d1.negative == d2.negative:
        if scientific:
            return d1.estring
        else:
            return d1.string

    if d1.negative == d2.negative and d1.digits == d2.digits:
        s1 = d1.s
        s2 = d2.s
        common_len = len(s1)
        for i, c in enumerate(s1):
            if c != s2[i]:
                common_len = i
                break
        common = s1[:common_len]
        rest1 = s1[common_len:]
        rest2 = s2[common_len:]

        if d1.negative:
            prefix = '-'
        else:
            prefix = ''
        prefix1 = ''
        prefix2 = ''

    else:
        common = ''
        rest1 = d1.s
        rest2 = d2.s

        prefix = ''
        if d1.negative:
            prefix1 = '-'
        else:
            prefix1 = '+'
        if d2.negative:
            prefix2 = '-'
        else:
            prefix2 = '+'

    if scientific:
        offset = len(common) + min(len(rest1), len(rest2)) - 1
        e = exp + offset
        exp = -offset
    else:
        e = None

    if exp >= 0:
        suffix = ('0' * exp) + '.'
        body = prefix + common + '[' + prefix1 + rest1 + suffix + '-' + prefix2 + rest2 + suffix + ']'

    elif common and -exp >= len(rest1):
        exp = exp + len(rest1)
        if exp == 0:
            body = prefix + common + '.' + '[' + rest1 + '-' + rest2 + ']'
        elif len(common) <= -exp:
            body = prefix + '.' + ('0' * -(len(common) + exp)) + common + '[' + rest1 + '-' + rest2 + ']'
        else:
            body = prefix + common[:exp] + '.' + common[exp:] + '[' + rest1 + '-' + rest2 + ']'

    else:
        if len(rest1) <= -exp:
            body1 = prefix1 + '.' + ('0' * -(len(rest1) + exp)) + rest1
        else:
            body1 = prefix1 + rest1[:exp] + '.' + rest1[exp:]

        if len(rest2) <= -exp:
            body2 = prefix2 + '.' + ('0' * -(len(rest2) + exp)) + rest2
        else:
            body2 = prefix2 + rest2[:exp] + '.' + rest2[exp:]

        body = prefix + common + '[' + body1 + '-' + body2 + ']'

    if e is None:
        return body
    else:
        if e >= 0:
            return body + 'e+' + str(e)
        else:
            return body + 'e' + str(e)


def digital_to_envelope(x):
    if x.is_zero():
        return (digital.Digital(x, c=1, exp=x.exp-1, negative=True, rounded=False),
                digital.Digital(x, c=1, exp=x.exp-1, negative=False, rounded=False))
    else:
        tmp = digital.Digital(x, c=x.c<<1, exp=x.exp-1, rounded=False)
        return tmp.prev_float(), tmp.next_float()

def envelope_encloses(x, d1, d2):
    d = Dec(x)
    env_lo, env_hi = digital_to_envelope(x)
    env_d1 = Dec(env_lo)
    env_d2 = Dec(env_hi)

    exp = min(d.exp, d1.exp, d2.exp, env_d1.exp, env_d2.exp)
    d = d.scale(exp)
    d1 = d1.scale(exp)
    d2 = d2.scale(exp)
    env_d1 = env_d1.scale(exp)
    env_d2 = env_d2.scale(exp)

    if d1.negative != d2.negative:
        assert d1.negative and env_d1.negative and (not d2.negative) and (not env_d2.negative)
        return d1.c <= env_d1.c and d2.c <= env_d2.c
    else:
        assert d1.negative == d2.negative == env_d1.negative == env_d2.negative and d1.c < d2.c
        return env_d1.c <= d1.c and d1.c < d.c and d.c < d2.c and d2.c <= env_d2.c

def digital_to_dec_range(x):
    env1, env2 = digital_to_envelope(x)
    d1 = Dec(env1)
    d2 = Dec(env2)

    for p in range(1, max(d1.digits, d2.digits) + 1):
        if x.is_zero():
            d1r = d1.round(p, direction=0)
            d2r = d2.round(p, direction=0)
        else:
            d1r = d1.round(p, direction=1)
            d2r = d2.round(p, direction=0)

        d1r, d2r = d1r.normalize_to(d2r)
        if d1r.negative != d2r.negative or d1r.c != d2r.c:
            attempt = dec_range_to_digital(d1r, d2r)
            if attempt is not None and attempt.c == x.c and attempt.exp == x.exp and (attempt.negative == x.negative or x.is_zero()):
                return d1r, d2r

    raise ValueError('failed to find a dec range for {}'.format(repr(x)))

def dec_range_to_digital(d1, d2):
    d1, d2 = d1.normalize_to(d2)
    exp = d1.exp

    if exp >= 0:
        p1 = digital.Digital(c=int(d1.c * (_mpz_10 ** exp)))
        p2 = digital.Digital(c=int(d2.c * (_mpz_10 ** exp)))
        q = digital.Digital(c=1)
    else:
        p1 = digital.Digital(c=d1.c)
        p2 = digital.Digital(c=d2.c)
        q = digital.Digital(c=int(_mpz_10 ** -exp))

    mp1 = digital_to_mpfr(p1)
    mp2 = digital_to_mpfr(p2)
    mq = digital_to_mpfr(q)

    if d1.negative != d2.negative:
        if p1.is_zero() and p2.is_zero():
            raise ValueError('empty dec range {} - {}'.format(repr(d1), repr(d2)))
        if d2.negative:
            d1, d2 = d2, d1

        # zero case
        uncertain = True
        candidate = None
        current = digital.Digital(m=0, exp=0)

        while uncertain:
            if envelope_encloses(current, d1, d2):
                candidate = current
                current = current.prev_float()
            else:
                if candidate is not None:
                    return candidate
                else:
                    current = current.next_float()

    else:
        if d1.c == d2.c:
            raise ValueError('empty dec range {} - {}'.format(repr(d1), repr(d2)))
        if d2.c < d1.c:
            d1, d2 = d2, d1

        if d1.negative:
            sign = -1
        else:
            sign = 1

        # nonzero case
        prec = 1
        uncertain = True
        candidate = None
        failures = 0

        while uncertain:
            with gmp.context(
                    # parameters stolen from compute
                    precision=prec + 1,
                    emin=gmp.get_emin_min(),
                    emax=gmp.get_emax_max(),
                    subnormalize=False,
                    trap_underflow=True,
                    trap_overflow=True,
                    trap_inexact=False,
                    trap_invalid=False,
                    trap_erange=False,
                    trap_divzero=False,
                    round=gmp.RoundToZero,
            ) as gmpctx:
                f1 = (mp1 / mq) * sign
                f2 = (mp2 / mq) * sign

            if f1 == f2:
                prec += 1
                continue

            r1 = mpfr_to_digital(f1).round_m(max_p=prec)
            r2 = mpfr_to_digital(f2).round_m(max_p=prec)

            found_candidate = False

            # is it necessary to try everything?
            for r in [r2, r2.prev_float(), r2.next_float(), r1, r1.next_float(), r1.prev_float()]:
                if envelope_encloses(r, d1, d2):
                    candidate = r
                    found_candidate = True
                    break

            if not found_candidate:
                failures += 1

            # It's possible that at some precision with no candidates, the next (finer)
            # precision has a candidate, due to the way the envelope boundaries line up.
            # So, continue checking finer precisions until we have more than two in a row
            # with no candidates.
            if failures > 3:
                uncertain = False
            else:
                prec += 1

        # some intervals don't actually have a (nonzero) candidate, if one side is too close to zero
        return candidate

        # if candidate is None:
        #     raise ValueError('failed to find an enclosing candidate for {}, {}'.format(repr(d1), repr(d2)))
        # else:
        #     return candidate



def nearest_uncertain_to_digital(negative, d1, d2):
    #print(d1, d2)

    c1, tens1 = d1
    c2, tens2 = d2

    # normalize exponent
    tens = min(tens1, tens2)
    if tens1 > tens:
        c1 *= 10 ** (tens1 - tens)
    if tens2 > tens:
        c2 *= 10 ** (tens2 - tens)

    if c1 == c2:
        # they're the same, nothing to see here...
        return None

    if tens >= 0:
        p1 = digital.Digital(c=c1 * (10**tens))
        p2 = digital.Digital(c=c2 * (10**tens))
        q = digital.Digital(c=1)
    else:
        p1 = digital.Digital(c=c1)
        p2 = digital.Digital(c=c2)
        q = digital.Digital(c=10**-tens)

    mp1 = digital_to_mpfr(p1)
    mp2 = digital_to_mpfr(p2)
    mq = digital_to_mpfr(q)

    # TODO: this is super slow!!! (like linear in p)
    prec = 2
    uncertain = True
    while uncertain:
        with gmp.context(
                # mostly stolen from compute, except:
                # don't use any extra precision
                precision=prec,
                emin=gmp.get_emin_min(),
                emax=gmp.get_emax_max(),
                subnormalize=False,
                trap_underflow=True,
                trap_overflow=True,
                trap_inexact=False,
                trap_invalid=False,
                trap_erange=False,
                trap_divzero=False,
                # round nearest; we're using MPRF's rounding directly
                # without fixing it up, and I think this makes
                # the most sense when studying the envelopes
                round=gmp.RoundToNearest,
        ) as gmpctx:
            f1 = mp1 / mq
            f2 = mp2 / mq

        r1 = mpfr_to_digital(f1)
        r2 = mpfr_to_digital(f2)

        r1c = r1.c
        r2c = r2.c
        # normalize exponent again
        rexp = min(r1.exp, r2.exp)
        if r1.exp > rexp:
            r1c <<= (r1.exp - rexp)
        if r2.exp > rexp:
            r2c <<= (r2.exp - rexp)

        rdiff = abs(r1c - r2c)
        #print(prec, f1, f2, rdiff, r1c, r2c)
        if rdiff == 2 and (r1c & 1 == 1) and (r2c & 1 == 1):
            if r1c > r2c:
                rc = r1c >> 1
            else:
                rc = r2c >> 1

            return digital.Digital(negative=negative, c=rc, exp=rexp+1)
        # TODO not actually sure if this always works yolo
        elif rdiff > 25:
            uncertain = False
        else:
            prec += 1

    # try again, assuming a power of two?
    # In that case, we need to parse the smaller magnitude side with 1 bit less precision
    prec = 3
    uncertain = True
    while uncertain:
        # larger
        with gmp.context(
                # mostly stolen from compute, except:
                # don't use any extra precision
                precision=prec,
                emin=gmp.get_emin_min(),
                emax=gmp.get_emax_max(),
                subnormalize=False,
                trap_underflow=True,
                trap_overflow=True,
                trap_inexact=False,
                trap_invalid=False,
                trap_erange=False,
                trap_divzero=False,
                # round nearest; we're using MPRF's rounding directly
                # without fixing it up, and I think this makes
                # the most sense when studying the envelopes
                round=gmp.RoundToNearest,
        ) as gmpctx:
            if c1 > c2:
                f1 = mp1 / mq
            else:
                f2 = mp2 / mq
        # smaller
        with gmp.context(
                # mostly stolen from compute, except:
                # don't use any extra precision
                precision=prec - 1,
                emin=gmp.get_emin_min(),
                emax=gmp.get_emax_max(),
                subnormalize=False,
                trap_underflow=True,
                trap_overflow=True,
                trap_inexact=False,
                trap_invalid=False,
                trap_erange=False,
                trap_divzero=False,
                # round nearest; we're using MPRF's rounding directly
                # without fixing it up, and I think this makes
                # the most sense when studying the envelopes
                round=gmp.RoundToNearest,
        ) as gmpctx:
            if c1 > c2:
                f2 = mp2 / mq
            else:
                f1 = mp1 / mq

        r1 = mpfr_to_digital(f1)
        r2 = mpfr_to_digital(f2)

        r1c = r1.c
        r2c = r2.c
        # normalize exponent again
        rexp = min(r1.exp, r2.exp)
        if r1.exp > rexp:
            r1c <<= (r1.exp - rexp)
        if r2.exp > rexp:
            r2c <<= (r2.exp - rexp)

        rdiff = abs(r1c - r2c)
        #print(prec, f1, f2, rdiff, r1c, r2c)
        if rdiff == 2 and (r1c & 1 == 1) and (r2c & 1 == 1):
            if r1c > r2c:
                rc = r1c >> 1
            else:
                rc = r2c >> 1

            return digital.Digital(negative=negative, c=rc, exp=rexp+1)
        # TODO not actually sure if this always works yolo
        elif rdiff > 25:
            uncertain = False
        else:
            prec += 1

    # these values don't seem to correspond to an nearest uncertain string: bail out
    return None


def digital_to_nearest_uncertain_string(x):
    if x.is_zero():
        return '[0]'
    else:
        next_digital = digital.Digital(negative=x.negative, c=(x.c << 1) + 1, exp=x.exp - 1)
        prev_digital = digital.Digital(negative=x.negative, c=(x.c << 1) - 1, exp=x.exp - 1)

        next_c10, next_e10 = decimal_expansion(next_digital)
        prev_c10, prev_e10 = decimal_expansion(prev_digital)

        maxd = max(len(str(next_c10)), len(str(prev_c10)))

        # TODO this is also super slow!!! (like linear in number of digits...)
        for prec in range(1, maxd + 2): # TODO + 2 yolo
            for d2 in round_dec((next_c10, next_e10), prec):
                for d1 in round_dec((prev_c10, prev_e10), prec):
                    if (d1 != d2):
                        attempt = nearest_uncertain_to_digital(x.negative, d1, d2)
                        if attempt is not None and attempt.p == x.p and attempt == x:
                            # we found one that works
                            c1, tens1 = d1
                            c2, tens2 = d2
                            # normalize
                            tens = min(tens1, tens2)
                            if tens1 > tens:
                                c1 *= 10 ** (tens1 - tens)
                            if tens2 > tens:
                                c2 *= 10 ** (tens2 - tens)
                            # stringulate
                            s1 = dec_to_str((c1, tens))
                            s2 = dec_to_str((c2, tens))
                            if x.negative:
                                prefix = '-'
                            else:
                                prefix = ''
                            return prefix + combine_dec_strings(s1, s2)

        print('failed for ' + str(x))
        print(repr(digital_to_mpfr(prev_digital)))
        print(prev_c10, prev_e10)
        print(repr(prev_digital))
        print(repr(digital_to_mpfr(next_digital)))
        print(next_c10, next_e10)
        print(repr(next_digital))
        return None


def combine_dec_strings(s1, s2):
    i = 0
    for c1, c2 in zip(s1, s2):
        if c1 == c2:
            i += 1
        else:
            break
    return s1[:i] + '[' + s1[i:] + '-' + s2[i:] + ']'


# TODO: doesn't work for zero yet!
_nearest_uncertain_re = re.compile(r'([-+]?)([.0-9]*)\[([.0-9]+)-([.0-9]+)\](?:[eE]([-+]?[0-9]+))?')
def nearest_uncertain_string_to_digital(s):
    if s == '[0]':
        return digital.Digital(m=0)

    m = _nearest_uncertain_re.fullmatch(s)
    if m is None:
        raise ValueError('invalid literal for uncertain digital: {}'.format(s))

    if m.group(1) == '-':
        negative = True
    else:
        negative = False

    s1 = m.group(2) + m.group(3)
    s2 = m.group(2) + m.group(4)
    if s1.count('.') > 1 or s2.count('.') > 1:
        raise ValueError('invalid literal for uncertain digital: {}'.format(s))

    if m.group(5) is None:
        etens = 0
    else:
        etens = int(m.group(4))

    c1, tens1 = str_to_dec(s1)
    tens1 += etens
    c2, tens2 = str_to_dec(s2)
    tens2 += etens

    return nearest_uncertain_to_digital(negative, (c1, tens1), (c2, tens2))


def arith_sim(a, b):
    """Compute the 'arithmetic bit similarity' between a and b, defined as:
                  | a - b |
        -log2( --------------- )
               min( |a|, |b| )
    That is to say, arithmetic similarity is the negative log base 2 of the
    relative difference between a and b, with reference to whichever has
    smaller magnitude. For positive results, this is roughly an upper bound
    on the number of binary digits that are the same between the two numbers;
    for negative results, it is roughly the negative magnitude of the difference
    in exponents.
    """

    prec = max(53, 1 + max(a.e, b.e) - min(a.n, b.n))

    mpfr_a = digital_to_mpfr(a)
    mpfr_b = digital_to_mpfr(b)

    if gmp.is_nan(mpfr_a) or gmp.is_nan(mpfr_b):
        return float('nan')

    if mpfr_a == mpfr_b:
        return float('inf')

    with gmp.context(
            precision=prec,
            emin=gmp.get_emin_min(),
            emax=gmp.get_emax_max(),
            trap_underflow=True,
            trap_overflow=True,
            trap_inexact=True,
            trap_invalid=True,
            trap_erange=False,
            trap_divzero=True,
    ):
        diff = abs(mpfr_a - mpfr_b)

    with gmp.context(
            precision=prec,
            emin=gmp.get_emin_min(),
            emax=gmp.get_emax_max(),
            trap_underflow=True,
            trap_overflow=True,
            trap_inexact=False,
            trap_invalid=True,
            trap_erange=False,
            trap_divzero=False,
    ):
        reldiff = diff / min(abs(mpfr_a), abs(mpfr_b))

    with gmp.ieee(64):
        sim = -gmp.log2(reldiff)

    return float(sim)


def geo_sim(a, b):
    """Compute the 'geometric bit similarity' between a and b, defined as:
               |        a    |
        -log2( | log2( --- ) | )
               |        b    |
    That is to say, geometric similarity is the negative log base 2 of the
    magnitude of the log base 2 of the ratio a / b. For positive results, this
    is roughly an upper bound on the number of binary digits that are the same
    between the numbers; for negative results, it is roughtly the negative magnitude
    of the number of bits that are different between the exponents.

    In general, the geometric similarity is probably more useful when trying
    to interpret fractional values, though for positive results, the floors of the
    arithmetic and geometric similarities will usually agree.

    This measure is the same as John Gustafson's "decimal accuracy," as defined
    in https://posithub.org/docs/Posits4.pdf, section 7.4.
    """
    prec = max(53, 1 + max(a.e, b.e) - min(a.n, b.n))

    mpfr_a = digital_to_mpfr(a)
    mpfr_b = digital_to_mpfr(b)

    if gmp.is_nan(mpfr_a) or gmp.is_nan(mpfr_b):
        return float('nan')

    if mpfr_a == 0 and mpfr_b == 0:
        return float('inf')
    elif mpfr_a == 0 or mpfr_b == 0:
        return float('-inf')

    with gmp.context(
            precision=prec,
            emin=gmp.get_emin_min(),
            emax=gmp.get_emax_max(),
            trap_underflow=True,
            trap_overflow=True,
            trap_inexact=False,
            trap_invalid=True,
            trap_erange=True,
            trap_divzero=False,
    ):
        ratio = mpfr_a / mpfr_b
        if ratio <= 0:
            return float('-inf')
        reldiff = abs(gmp.log2(ratio))

    with gmp.ieee(64):
        sim = -gmp.log2(reldiff)

    return float(sim)

def geo_sim10(a, b):
    """Compute the 'decimals of accuracy' between a and b, defined as:
                |        a    |
        -log10( | log10( --- ) | )
                |        b    |

    """
    prec = max(53, 1 + max(a.e, b.e) - min(a.n, b.n))

    mpfr_a = digital_to_mpfr(a)
    mpfr_b = digital_to_mpfr(b)

    if gmp.is_nan(mpfr_a) or gmp.is_nan(mpfr_b):
        return float('nan')

    if mpfr_a == 0 and mpfr_b == 0:
        return float('inf')
    elif mpfr_a == 0 or mpfr_b == 0:
        return float('-inf')

    with gmp.context(
            precision=prec,
            emin=gmp.get_emin_min(),
            emax=gmp.get_emax_max(),
            trap_underflow=True,
            trap_overflow=True,
            trap_inexact=False,
            trap_invalid=True,
            trap_erange=True,
            trap_divzero=False,
    ):
        ratio = mpfr_a / mpfr_b
        if ratio <= 0:
            return float('-inf')
        reldiff = abs(gmp.log10(ratio))

    with gmp.ieee(64):
        sim = -gmp.log10(reldiff)

    return float(sim)


# deprecated


def withnprec(op, *args, min_n = -1075, max_p = 53,
               emin = gmp.get_emin_min(), emax = gmp.get_emax_max()):
    """Compute op(*args), with n >= min_n and precision <= max_p.

    Arguments are provided as mpfrs; they are treated as exact values, so their
    precision is unimportant except where it affects their value. The result is a
    Titanic Sink.

    The requested precision can be less than gmpy2's minimum of 2 bits, and zeros
    produced by rounding should have all the right properties.

    TODO: the arguments themselves should not be zero; for computations involving
    zero, special cases should be written per operation.

    TODO: the output is correctly rounded under RNE rules; others should probably
    be supported.
    """

    if max_p < 0:
        raise ValueError('cannot compute a result with less than 0 max precision, got {}'.format(repr(max_p)))

    # The precision we want to compute with is at least p+1, to determine RNE behavior from RTZ.
    # We use max_p + 2 to ensure the quantity is at least 2 for mpfr.
    prec = max_p + 2

    # This context allows us to tolerate inexactness, but no other surprising behavior. Those
    # cases should be handled explicitly per operation.
    with gmp.context(
            precision=prec,
            emin=emin,
            emax=emax,
            trap_underflow=True,
            trap_overflow=True,
            trap_inexact=False,
            trap_invalid=True,
            trap_erange=True,
            trap_divzero=True,
            # IMPORTANT: we need RTZ in order to be able to multiple-round accurately
            # to the desired precision.
            round=gmp.RoundToZero,
    ) as gmpctx:
        candidate = op(*args)
        op_inexact = gmpctx.inexact

    m, exp = conversion.mpfr_to_mantissa_exp(candidate)

    # Now we need to round to the correct number of bits

    mbits = m.bit_length()

    n = exp - 1
    e = n + mbits
    target_n = max(e - max_p, min_n)
    xbits = target_n - n

    # Split the result into 3 components: sign, significant bits (rounded down), and half bit

    negative = conversion.is_neg(candidate)
    c = abs(m)

    if c > 0:
        sig = c >> xbits
        half_x = c & bitmask(xbits)
        half = half_x >> (xbits - 1)
        x = half_x & bitmask(xbits - 1)
    else:
        sig = 0
        half_x = 0
        half = 0
        x = 0

    # Now we need to decide how to round. The value we have in sig was rounded toward zero, so we
    # look at the half bit and the inexactness of the operation to decide if we should round away.

    if half > 0:
        # greater than halfway away implied by inexactness, or demonstrated by nonzero xbits
        if x > 0 or op_inexact:
            # if we have no precision, round away by increasing n
            if max_p == 0:
                target_n += 1
            else:
                sig += 1
        # TODO: hardcoded RNE
        elif sig & 1 > 0:
            sig += 1

    # fixup extra precision from a carry out
    # TODO in theory this could all be abstracted away with .away()
    if sig.bit_length() > max_p:
        # TODO assert
        if sig & 1 > 0:
            raise AssertionError('cannot fixup extra precision: {} {}'.format(repr(op), repr(args))
                                 + '  {}, m={}, exp={}, decoded as {} {} {} {}'.format(
                                     repr(candidate), repr(m), repr(exp), negative, sig, half, x))
        sig >>= 1
        target_n += 1

    result_inexact = half > 0 or x > 0 or op_inexact
    result_sided = sig == 0 and not result_inexact

    return Sink(c=sig,
                exp=target_n + 1,
                negative=negative,
                inexact=result_inexact,
                sided=result_sided,
                full=False)


# All these operations proceed exactly with the bits that they are given,
# which means that they can produce results that are more precise than should
# be allowed given the inexactness of their inputs. The idea is not to call them
# with unacceptably precise rounding specifications.

# These operations won't crash if given inexact zeros, but nor will they do anything
# clever with the precision / exponent of the result.

# For zeros that are computed from non-zero inputs, the precision / exponent
# should be reasonable.

# Though extra precision may be present, the inexact flags of the results should
# always be set correctly.

# Actually, do these wrappers actually do anything? In all cases it seems like
# we just want a computed zero to have the specified min_n...

def add(x, y, min_n = -1075, max_p = 53):
    """Add two sinks, rounding according to min_n and max_p.
    TODO: rounding modes
    """
    result = withnprec(gmp.add, x.to_mpfr(), y.to_mpfr(),
                       min_n=min_n, max_p=max_p)

    inexact = x.inexact or y.inexact or result.inexact

    #TODO technically this could do clever things with the interval
    return Sink(result, inexact=inexact, full=False, sided=False)


def sub(x, y, min_n = -1075, max_p = 53):
    """Subtract two sinks, rounding according to min_n and max_p.
    TODO: rounding modes
    """
    result = withnprec(gmp.sub, x.to_mpfr(), y.to_mpfr(),
                       min_n=min_n, max_p=max_p)

    inexact = x.inexact or y.inexact or result.inexact

    #TODO technically this could do clever things with the interval
    return Sink(result, inexact=inexact, full=False, sided=False)


def mul(x, y, min_n = -1075, max_p = 53):
    """Multiply two sinks, rounding according to min_n and max_p.
    TODO: rounding modes
    """

    # special case for the exponent of zero
    if x.is_zero() or y.is_zero():
        e = x.e + y.e
        return Sink(c = 0,
                    exp = e + 1, # since for 0, e = n, and n = exp - 1
                    negative = x.negative != y.negative,
                    inexact = not (x.is_exactly_zero() or y.is_exactly_zero()),
                    # TODO interval stuff
                    sided = False,
                    full = False)

    result = withnprec(gmp.mul, x.to_mpfr(), y.to_mpfr(),
                       min_n=min_n, max_p=max_p)

    inexact = x.inexact or y.inexact or result.inexact

    #TODO technically this could do clever things with the interval
    return Sink(result, inexact=inexact, full=False, sided=False)


def div(x, y, min_n = -1075, max_p = 53):
    """Divide to sinks x / y, rounding according to min_n and max_p.
    TODO: rounding modes
    """

    # special case for the exponent of zero
    if x.is_zero():
        e = x.e - y.e
        return Sink(c = 0,
                    exp = e + 1, # since for 0, e = n, and n = exp - 1
                    negative = x.negative != y.negative,
                    inexact = not x.is_exactly_zero(),
                    # TODO interval stuff
                    sided = False,
                    full = False)

    elif y.is_zero():
        raise ZeroDivisionError('division by zero: {} / {}'.format(repr(x), repr(y)))

    result = withnprec(gmp.div, x.to_mpfr(), y.to_mpfr(),
                       min_n=min_n, max_p=max_p)

    inexact = x.inexact or y.inexact or result.inexact

    #TODO technically this could do clever things with the interval
    return Sink(result, inexact=inexact, full=False, sided=False)


def sqrt(x, min_n = -1075, max_p = 53):
    """Take the square root of a sink, rounding according to min_n and max_p.
    TODO: rounding modes
    """
    result = withnprec(gmp.sqrt, x.to_mpfr(),
                       min_n=min_n, max_p=max_p)

    inexact = x.inexact or result.inexact

    #TODO technically this could do clever things with the interval
    return Sink(result, inexact=inexact, full=False, sided=False)


def floor(x, min_n = -1075, max_p = 53):
    """Take the floor of x, rounding according to min_n and max_p.
    TODO: rounding modes
    """
    result = withnprec(gmp.floor, x.to_mpfr(),
                       min_n=min_n, max_p=max_p)

    # TODO it's noot 100% clear who should do the checking here.
    # Technically, any exactly computed floor is exact unless
    # its n permits an ulp variance of at least unity.

    # However, implementations may want to decide that certain floors
    # are inexact even though the representation does not require them
    # to be so, i.e. floor(1.00000000000~) could be 0 or 1 depending
    # on which way we rounded, even though at that precision ulps are
    # relatively small.

    inexact = (x.inexact or result.inexact) and min_n >= 0

    #TODO technically this could do clever things with the interval
    return Sink(result, inexact=inexact, full=False, sided=False)


def fmod(x, y, min_n = -1075, max_p = 53):
    """Compute the remainder of x mod y, rounding according to min_n and max_p.
    TODO: rounding modes
    """
    result = withnprec(gmp.fmod, x.to_mpfr(), y.to_mpfr(),
                       min_n=min_n, max_p=max_p)

    inexact = x.inexact or y.inexact or result.inexact

    #TODO technically this could do clever things with the interval
    return Sink(result, inexact=inexact, full=False, sided=False)


def pow(x, y, min_n = -1075, max_p = 53):
    """Raise x ** y, rounding according to min_n and max_p.
    TODO: rounding modes
    """
    result = withnprec(lambda x, y: x**y, x.to_mpfr(), y.to_mpfr(),
                       min_n=min_n, max_p=max_p)

    inexact = x.inexact or y.inexact or result.inexact

    #TODO technically this could do clever things with the interval
    return Sink(result, inexact=inexact, full=False, sided=False)


def sin(x, min_n = -1075, max_p = 53):
    """Compute sin(x), rounding according to min_n and max_p.
    TODO: rounding modes
    """
    result = withnprec(gmp.sin, x.to_mpfr(),
                       min_n=min_n, max_p=max_p)

    inexact = x.inexact or result.inexact

    #TODO technically this could do clever things with the interval
    return Sink(result, inexact=inexact, full=False, sided=False)


# helpers to produce some useful constants

def pi(p):
    # TODO no support for rounding modes
    if p < 0:
        raise ValueError('precision must be at least 0')
    elif p == 0:
        # TODO is this right???
        return Sink(m=0, exp=3, inexact=True, sided=True, full=False)
    elif p == 1:
        return Sink(m=1, exp=2, inexact=True, sided=False, full=False)
    else:
        with gmp.context(
                    precision=p,
                    emin=-1,
                    emax=2,
                    trap_underflow=True,
                    trap_overflow=True,
                    trap_inexact=False,
                    trap_invalid=True,
                    trap_erange=True,
                    trap_divzero=True,
        ):
            x = gmp.const_pi()
    return Sink(x, inexact=True, sided=False, full=False)
