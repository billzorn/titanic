# Real numbers, sort of. An IEEE 754 compliant implementation.

# Floating point numbers are usually used in place of the reals,
# but are themselves always rational. The purpose of this library
# is to track values and expressions involving real numbers in
# a symbolic way, that agrees with the IEEE notions of inf, nan,
# and -0, and that can be deterministically compared to a rational
# to search for the nearest floating point number in some representation.

# The current implementation is built on top of sympy. For the meaning of
# operations on inputs with values not in the real reals, see the IEEE 754
# spec or the implementation.

import sympy
Rational = sympy.Rational
symbolic_zero = sympy.sympify(0)

# It turns out comparing numbers is rather hard.

default_maxn = 1000000
default_n = 100
default_n_scale = 10
def default_simplify(x):
    return sympy.simplify(x)

class ConjectureEqualityException(Exception):
    def __init__(self, a, b, n):
        self.a = a
        self.b = b
        self.n = n

    def __str__(self):
        return ('conjecture: {} == {} (tested to {:d} decimal places)'
                .format(str(self.a), str(self.b), self.n))

    def __repr__(self):
        return ('{}({}, {}, {:d})'
                .format(type(self).__name__, repr(self.a), repr(self.b), self.n))

def eval_until_at_least(x, prec,
                        n = default_n, maxn = default_maxn, scale = default_n_scale,
                        abort_incomparables = True):
    assert x.is_real and x.is_finite
    assert isinstance(n, int)
    assert isinstance(maxn, int)
    assert 0 < n and n < maxn
    assert isinstance(scale, int)
    assert scale >= 2
    assert abort_incomparables is True or abort_incomparables is False

    # f = x.evalf(prec, maxn=n)
    # if f.is_comparable:
    #     return f
    # # is it better to just throw in maxn instead of a loop? or does that waste resources?
    # else:
    #     while n < maxn:
    #         n = n * default_n_scale
    #         f = x.evalf(prec, maxn=n)
    #         if f.is_comparable:
    #             return f
    #     if abort_incomparables:
    #         raise ConjectureEqualityException(x, symbolic_zero, n)
    #     else:
    #         return f

    # let's try without the loop
    f = x.evalf(prec, maxn=maxn)
    if f.is_comparable:
        return f
    if abort_incomparables:
        raise ConjectureEqualityException(x, symbolic_zero, n)
    else:
        return f


def decide_order(x, y):
    assert x.is_real and x.is_finite
    assert y.is_real and y.is_finite

    z = default_simplify(x - y)
    if z.is_rational:
        comp = z
    else:
        comp = eval_until_at_least(z, 2)
    assert comp.is_comparable

    if comp.is_zero:
        return 0
    elif comp.is_positive:
        return 1
    elif comp.is_negative:
        return -1
    else:
        raise ValueError('unable to decide order of {} and {}'
                         .format(repr(x), repr(y)))

# Input and parsing.

import re

inf_strs_re = re.compile(r'inf|infinity|oo')
nan_strs_re = re.compile(r's?nan(([-0-9]+)|\(([-0-9]+)\))?')
preferred_inf_str = 'inf'
preferred_nan_str = 'nan'

exp_strs_re = re.compile(r'([0-9.]+)\*(([0-9]+)\*\*([-0-9]+)|\(([0-9]+)\*\*([-0-9]+)\))')
# >>> exp_str_re.fullmatch('1.015625*(2**-13)').groups()
# ('1.015625', '(2**-13)', None, None, '2', '-13')
# >>> exp_str_re.fullmatch('1.015625*2**-13').groups()
# ('1.015625', '2**-13', '2', '-13', None, None)

# fpcore approved constants
math_constants = {
    'e'        : sympy.E,
    'log2e'    : sympy.log(sympy.E, 2),
    'log10e'   : sympy.log(sympy.E, 10),
    'ln2'      : sympy.log(2),
    'ln10'     : sympy.log(10),
    'pi'       : sympy.pi,
    'pi_2'     : sympy.pi / 2,
    'pi_4'     : sympy.pi / 4,
    '1_pi'     : 1 / sympy.pi,
    '2_pi'     : 2 / sympy.pi,
    '2_sqrtpi' : 2 / sympy.sqrt(sympy.pi),
    'sqrt2'    : sympy.sqrt(2),
    'sqrt1_2'  : 1 / sympy.sqrt(2),
}
math_strs_re = re.compile(r'|'.join(re.escape(k) for k in math_constants))

# Affinely extended real numbers with signed zero and labeled NaN.

class FReal(object):

    def _sign(self):
        if self.negative is True:
            return -1
        else:
            return 1
    sign = property(_sign)

    def _isinf(self):
        return self.infinite is True
    isinf = property(_isinf)

    def _isnan(self):
        return self.infinite is False and self.magnitude is None
    isnan = property(_isnan)

    def _isrational(self):
        if self.infinite is True or self.magnitude is None:
            return False
        else:
            return bool(self.magnitude.is_rational)
    isrational = property(_isrational)

    def _iszero(self):
        if self.infinite is True or self.magnitude is None:
            return False
        elif self.isrational:
            return self.magnitude == 0
        else:
            return decide_order(self.magnitude, symbolic_zero) == 0
    iszero = property(_iszero)


    def _isinteger(self):
        if self.infinite is True or self.magnitude is None:
            return False
        else:
            return bool(self.magnitude.is_integer)
    isinteger = property(_isinteger)

    def _nan_payload(self):
        if self.isnan:
            return self.payload
        else:
            return None
    nan_payload = property(_nan_payload)

    def _rational_numerator(self):
        if self.isinf:
            return self.sign
        elif self.isnan:
            return 0
        elif self.isrational:
            return self.magnitude.p * self.sign
        else:
            return None
    rational_numerator = property(_rational_numerator)

    def _rational_denominator(self):
        if self.isinf or self.isnan:
            return 0
        elif self.isrational:
            return self.magnitude.q
        else:
            return None
    rational_denominator = property(_rational_denominator)

    # this will destroy the sign of zero
    def _symbolic_value(self):
        if self.isinf:
            return sympy.oo * self.sign
        elif self.isnan:
            return sympy.nan
        else:
            return self.magnitude * self.sign
    symbolic_value = property(_symbolic_value)

    # so will this
    def numeric_value(self, prec, maxn = default_maxn, abort_incomparables = True):
        if self.isinf or self.isnan:
            return self.symbolic_value.evalf()
        else:
            return eval_until_at_least(default_simplify(self.symbolic_value), prec,
                                       maxn=maxn, abort_incomparables=abort_incomparables)

    def _valid(self):
        assert self.negative is True or self.negative is False
        assert self.infinite is True or self.infinite is False

        # inf
        if self.infinite:
            assert self.magnitude is None
            assert self.payload == 0
        else:
            # nan
            if self.magnitude is None:
                assert self.payload != 0
            # real
            else:
                # magnitude is always positive
                assert self.magnitude.is_real
                # Some classes, like sympy.erf, seem to not support this.
                # This is a good test for sin(oo), but problematic for say erf(1).
                assert self.magnitude.is_finite
                if self.magnitude.is_rational:
                    assert self.magnitude >= 0
                else:
                    assert decide_order(self.magnitude, symbolic_zero) >= 0
                assert self.payload == 0

    _kw_negative = 'negative'
    _kw_infinite = 'infinite'
    _kw_payload = 'payload'

    def __init__(self, x = None, negative = None, infinite = False, payload = 1):
        assert negative is None or negative is True or negative is False
        assert infinite is True or infinite is False

        # Only nonzero for nans. This default will be overwritten with the
        # argument in the case of a nan.
        self.payload = 0

        # explicitly told to construct infinity
        if infinite:
            assert x is None
            self.magnitude = None
            # by default, an explicitly constructed infinity is positive
            self.negative = bool(negative)
            self.infinite = True

        # otherwise, figure out what to do with x
        else:
            self.infinite = False

            # explicitly told to construct nan
            if x is None:
                self.magnitude = None
                # by default, an explicitly constructed nan is positive
                self.negative = bool(negative)
                # use the given payload, which is > 0
                self.payload = payload
                assert self.payload != 0

            # an integer, with optional sign
            elif isinstance(x, int):
                if negative is None:
                    # it is ok to compare ints and rationals directly to zero
                    self.negative = bool(x < 0)
                    if self.negative:
                        self.magnitude = Rational(-x, 1)
                    else:
                        self.magnitude = Rational(x, 1)
                else:
                    assert x >= 0
                    self.magnitude = Rational(x, 1)
                    self.negative = negative

            # a rational number, with optional sign
            elif isinstance(x, Rational):
                if negative is None:
                    self.negative = bool(x < 0)
                    if self.negative:
                        self.magnitude = -x
                    else:
                        self.magnitude = x
                else:
                    assert x >= 0
                    self.magnitude = x
                    self.negative = negative

            # a string, which we have to parse
            elif isinstance(x, str):
                # try to get sign from the string
                s = x.strip().lower()
                if s.startswith('-'):
                    str_negative = True
                    s = s[1:]
                elif s.startswith('+'):
                    str_negative = False
                    s = s[1:]
                else:
                    str_negative = False

                # inf
                if inf_strs_re.fullmatch(s):
                    if negative is None:
                        self.negative = str_negative
                    else:
                        assert str_negative is False
                        self.negative = negative
                    self.magnitude = None
                    self.infinite = True
                # nan
                elif nan_strs_re.fullmatch(s):
                    if negative is None:
                        self.negative = str_negative
                    else:
                        assert str_negative is False
                        self.negative = negative
                    self.magnitude = None
                    # try to parse a payload from the string
                    nan_match = nan_strs_re.fullmatch(s)
                    if nan_match.group(2) is not None:
                        self.payload = int(nan_match.group(2))
                    elif nan_match.group(3) is not None:
                        self.payload = int(nan_match.group(3))
                    else:
                        self.payload = payload
                    assert self.payload != 0
                # known mathematical constants
                elif math_strs_re.fullmatch(s):
                    if negative is None:
                        self.negative = str_negative
                    else:
                        assert str_negative is False
                        self.negative = negative
                    self.magnitude = math_constants[s]
                # z3 exponent notation: '-1.015625*(2**-13)'
                elif exp_strs_re.fullmatch(s):
                    if negative is None:
                        self.negative = str_negative
                    else:
                        assert str_negative is False
                        self.negative = negative
                    exp_match = exp_strs_re.fullmatch(s)
                    c = Rational(exp_match.group(1))
                    if exp_match.group(3) is not None and exp_match.group(4) is not None:
                        b = Rational(exp_match.group(3))
                        e = Rational(exp_match.group(4))
                    else:
                        b = Rational(exp_match.group(5))
                        e = Rational(exp_match.group(6))
                    self.magnitude = c * (b ** e)
                # some other kind of string
                else:
                    try:
                        # The sign is tricky.
                        # Traditionally, -0 / 1 is -0, but -0 / -1 is 0.

                        # we may be missing the leading negation
                        r = Rational(s)
                        r_negative = bool(r < 0)
                        # so refactor it in with a logical xor, to ensure the correct sign for 0.
                        str_negative = str_negative != r_negative

                        if negative is None:
                            self.negative = str_negative
                            if r_negative:
                                self.magnitude = -r
                            else:
                                self.magnitude = r
                        else:
                            assert str_negative is False
                            self.negative = negative
                            self.magnitude = r

                    # If parsing as a rational fails, try again with original string.
                    # Note that sympification will always destroy the sign of 0; it can still
                    # be passed separately, if the expression is a positive magnitude.
                    except Exception:
                        r = sympy.sympify(x)
                        r = default_simplify(r)
                        assert r.is_real or r == sympy.nan
                        if r == sympy.nan:
                            str_negative = False
                        else:
                            # x might be some weird real number, so we have to be careful when comparing it
                            str_negative = decide_order(r, symbolic_zero) < 0

                        if negative is None:
                            self.negative = str_negative
                        else:
                            assert str_negative is False
                            self.negative = negative

                        if r.is_infinite:
                            self.magnitude = None
                            self.infinite = True
                        elif r == sympy.nan:
                            self.magnitude = None
                            self.payload = payload
                            assert self.payload != 0
                        else:
                            if str_negative:
                                self.magnitude = -r
                            else:
                                self.magnitude = r

            # otherwise, it must be some sympy object representing a real number or nan
            else:
                x = default_simplify(x)
                assert x.is_real or x == sympy.nan
                if x == sympy.nan:
                    x_negative = False
                else:
                    # x might be some weird real number, so we have to be careful when comparing it
                    x_negative = decide_order(x, symbolic_zero) < 0

                if negative is None:
                    self.negative = x_negative
                else:
                    assert x_negative is False
                    self.negative = negative

                if x.is_infinite:
                    self.magnitude = None
                    self.infinite = True
                elif x == sympy.nan:
                    self.magnitude = None
                    self.payload = payload
                    assert self.payload != 0
                else:
                    if x_negative:
                        self.magnitude = -x
                    else:
                        self.magnitude = x

        self._valid()

    def __str__(self):
        if self.sign < 0:
            sign_str = '-'
        else:
            sign_str = ''

        if self.isinf:
            return sign_str + preferred_inf_str
        elif self.isnan:
            return preferred_nan_str
        elif self.iszero:
            return sign_str + '0'
        elif self.isrational:
            if self.rational_denominator == 1:
                return str(self.rational_numerator)
            else:
                return str(self.rational_numerator) + '/' + str(self.rational_denominator)
        else:
            for k, x in math_constants.items():
                if self.magnitude == x:
                    return sign_str + k
            return str(self.magnitude * self.sign)

    def __repr__(self):
        my_class = type(self)

        if self.isinf:
            s = ', '.join((
                my_class._kw_negative + '=' + repr(self.negative),
                my_class._kw_infinite + '=' + repr(self.infinite),
            ))
        elif self.isnan:
            s = ', '.join((
                repr(None),
                my_class._kw_negative + '=' + repr(self.negative),
                my_class._kw_payload + '=' + repr(self.payload),
            ))
        elif self.iszero:
            s = ', '.join((
                repr(0),
                my_class._kw_negative + '=' + repr(self.negative),
            ))
        elif self.isrational:
            if self.rational_denominator == 1:
                s =  repr(self.rational_numerator)
            else:
                s = repr(repr(self.rational_numerator) + '/' + repr(self.rational_denominator))
        else:
            s = ''
            for k, x in math_constants.items():
                if self.magnitude == x:
                    if self.negative:
                        sign_str = '-'
                    else:
                        sign_str = ''
                    s = repr(sign_str + k)
                    break
            if s == '':
                s = repr(repr(self.magnitude * self.sign))

        return my_class.__name__ + '(' + s + ')'

    # arithmetic

    def __neg__(self):
        return FReal(self.magnitude, negative=not self.negative,
                     infinite=self.infinite, payload=self.payload)

    def __abs__(self):
        return FReal(self.magnitude, negative=False,
                     infinite=self.infinite, payload=self.payload)

    def __add__(self, x):
        if not isinstance(x, FReal):
            x = FReal(x)

        # inf and nan cases
        if self.isnan:
            return self
        elif x.isnan:
            return x
        elif self.isinf:
            if x.isinf:
                # same sign: return same infinity
                if self.sign == x.sign:
                    return self
                # opposite signs: return nan
                else:
                    # PARAM: what sign and payload do we use here?
                    return FReal(None, negative=self.negative, payload=1)
            else:
                return self
        elif x.isinf:
            return x

        # actually perform addition
        else:
            # same sign: add
            if self.sign == x.sign:
                m = self.magnitude + x.magnitude
                # Copying the sign from the left ensures that -0 + -0 = -0,
                # as required by the IEEE 754 standard.
                m_negative = self.negative
            else:
                # different signs: subtract according to which is which
                if self.sign < 0:
                    neg_m = self.magnitude
                    pos_m = x.magnitude
                else:
                    neg_m = x.magnitude
                    pos_m = self.magnitude

                comp = decide_order(neg_m, pos_m)

                if comp == -1:
                    m = pos_m - neg_m
                    m_negative = False
                elif comp == 0:
                    m = 0
                    # TODO: in RTN rounding, this sould produce -0 instead of 0.
                    m_negative = False
                else:
                    m = neg_m - pos_m
                    m_negative = True

            return FReal(m, negative=m_negative)

    def __radd__(self, y):
        if not isinstance(y, FReal):
            y = FReal(y)
        return y + self

    def __sub__(self, x):
        if not isinstance(x, FReal):
            x = FReal(x)
        return self + (-x)

    def __rsub__(self, y):
        if not isinstance(y, FReal):
            y = FReal(y)
        return y - self

    def __mul__(self, x):
        if not isinstance(x, FReal):
            x = FReal(x)

        # nan cases
        if self.isnan:
            return self
        elif x.isnan:
            return x
        else:
            # sign is always computed via this xor, unless an input or ouput is nan
            negative = self.sign != x.sign

            # infinities, might produce nan with 0
            if self.isinf:
                if x.iszero:
                    # PARAM: sign and payload for mult nans?
                    return FReal(None, negative=negative, payload=1)
                else:
                    return FReal(self.magnitude, negative=negative,
                                 infinite=self.infinite, payload=self.payload)
            elif x.isinf:
                if self.iszero:
                    # PARAM: sign and payload for mult nans?
                    return FReal(None, negative=negative, payload=1)
                else:
                    return FReal(x.magnitude, negative=negative,
                                 infinite=x.infinite, payload=x.payload)

            # actually multiply
            else:
                return FReal(self.magnitude * x.magnitude, negative=negative)

    def __rmul__(self, y):
        if not isinstance(y, FReal):
            y = FReal(y)
        return y * self

    def __truediv__(self, x):
        if not isinstance(x, FReal):
            x = FReal(x)

        # nan cases
        if self.isnan:
            return self
        elif x.isnan:
            return x
        else:
            # sign is always computed via this xor, unless an input or ouput is nan
            negative = self.sign != x.sign

            # infinities and zeros
            if self.isinf:
                if x.isinf:
                    # PARAM: sign and payload for inf/inf?
                    return FReal(None, negative=negative, payload=1)
                else:
                    return FReal(self.magnitude, negative=negative,
                                 infinite=self.infinite, payload=self.payload)
            elif self.iszero:
                if x.iszero:
                    # PARAM: sign and payload for 0/0?
                    return FReal(None, negative=negative, payload=1)
                else:
                    return FReal(self.magnitude, negative=negative,
                                 infinite=self.infinite, payload=self.payload)
            elif x.isinf:
                return FReal(0, negative=negative)
            elif x.iszero:
                return FReal(infinite=True, negative=negative)

            # actually divide
            else:
                return FReal(self.magnitude / x.magnitude, negative=negative)

    def __rtruediv__(self, y):
        if not isinstance(y, FReal):
            y = FReal(y)
        return y / self

    # only integer powers for now
    def __pow__(self, x):
        if isinstance(x, int):
            return self.pown(x)
        elif not isinstance(x, FReal):
            x = FReal(x)

        if x.isinteger:
            i = int(x.magnitude) * x.sign
            return self.pown(i)
        else:
            raise ValueError('only integer powers are supported; got: {}'.format(repr(x)))

    def __rpow__(self, y):
        if not isinstance(y, FReal):
            y = FReal(y)
        return y ** self

    # IEEE recommended functions

    def pown(self, i):
        assert isinstance(i, int)

        # Yep, the IEEE standard allows this to turn nan into 1.
        if i == 0:
            return FReal(1)
        elif self.isnan:
            return self
        else:
            # odd powers retain sign
            if i % 2 != 0:
                negative = self.negative
            # even powers are positive
            else:
                negative = False

            if self.iszero:
                if i < 0:
                    return FReal(negative=negative, infinite=True)
                else: # i > 0
                    return FReal(0, negative=negative)
            else:
                return FReal(self.magnitude ** i, negative=negative)

    # comparison

    # General comparison, according to IEEE guidelines.
    # For a.compareto(b), we will return:
    #    -1 iff a < b
    #     0 iff a = b
    #     1 iff a > b
    #  None iff a and b are unordered (i.e. at least one is NaN)
    def compareto(self, x):
        if not isinstance(x, FReal):
            x = FReal(x)

        # nan is unordered
        if self.isnan or x.isnan:
            return None
        # zeros are equal regardless of sign
        elif self.iszero and x.iszero:
            return 0
        # decide based purely on different sign
        elif self.sign == -1 and x.sign == 1:
            return -1
        elif self.sign == 1 and x.sign == -1:
            return 1
        # hard case is same sign... can we simplify this at all?
        elif self.sign == -1 and x.sign == -1:
            if self.isinf:
                if x.isinf:
                    return 0
                else:
                    return -1
            elif x.isinf:
                return 1
            else:
                return decide_order(x.magnitude, self.magnitude)
        else: # self.sign == 1 and x.sign == 1
            if self.isinf:
                if x.isinf:
                    return 0
                else:
                    return 1
            elif x.isinf:
                return -1
            else:
                return decide_order(self.magnitude, x.magnitude)

    # These are effectively unordered-quiet predicates, except
    # __ne__ is an unordered-quiet negation.

    def __lt__(self, x):
        comp = self.compareto(x)
        return comp is not None and comp < 0

    def __le__(self, x):
        comp = self.compareto(x)
        return comp is not None and comp <= 0

    def __eq__(self, x):
        comp = self.compareto(x)
        return comp == 0

    def __ne__(self, x):
        comp = self.compareto(x)
        return comp != 0

    def __ge__(self, x):
        comp = self.compareto(x)
        return comp is not None and 0 <= comp

    def __gt__(self, x):
        comp = self.compareto(x)
        return comp is not None and 0 < comp
