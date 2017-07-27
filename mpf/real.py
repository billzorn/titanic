# Ideally, this would use constructive reals or a CAS to deal with irrational
# numbers. For now, it's just rationals.

# It's also not really reals, as these reals can be inf or nan, so it's more
# like floating-point specific extended rationals. Whatever.

# import fractions
# Rational = fractions.Fraction

import sympy
Rational = sympy.Rational

inf_strs = {'inf', 'infinity', 'oo'}
nan_strs = {'nan'}
preferred_inf_str = 'inf'
preferred_nan_str = 'nan'

# _posinf = float('+inf')
# _neginf = float('-inf')
# _nan = float('nan')
# def _isnan(x):
#     return x != x

_posinf = sympy.oo
_neginf = -sympy.oo
_nan = sympy.nan
def _isnan(x):
    return x == _nan

class Real(object):

    def _isinf(self):
        return self.v == _posinf or self.v == _neginf
    isinf = property(_isinf)

    def __isnan(self):
        return _isnan(self.v)
    isnan = property(__isnan)

    def _numerator(self):
        if self.v == _posinf:
            return 1
        elif self.v == _neginf:
            return -1
        elif _isnan(self.v):
            return 0
        else:
            # return self.v.numerator
            return self.v.p
    numerator = property(_numerator)

    def _denominator(self):
        if self.v == _posinf:
            return 0
        elif self.v == _neginf:
            return 0
        elif _isnan(self.v):
            return 0
        else:
            # return self.v.denominator
            return self.v.q
    denominator = property(_denominator)

    def __init__(self, x):
        if isinstance(x, Rational):
            v = x

        elif isinstance(x, int):
            v = Rational(x, 1)

        elif isinstance(x, str):
            s = x.strip().lower()
            if s.startswith('-'):
                positive = False
                s = s[1:]
            elif s.startswith('+'):
                positive = True
                s = s[1:]
            else:
                positive = True

            if s in inf_strs:
                if positive:
                    v = _posinf
                else:
                    v = _neginf
            elif s in nan_strs:
                v = _nan
            else:
                v = Rational(x)

        else:
            if (x == _posinf) or (x == _neginf) or (_isnan(x)):
                v = x
            else:
                raise ValueError('can only make reals from placeholder floats inf, -inf, and nan, got: {}'.format(repr(x)))

        self.v = v

    def __str__(self):
        if self.denominator == 1:
            return str(self.numerator)
        else:
            return str(self.numerator) + '/' + str(self.denominator)

    def __repr__(self):
        if self.v == _posinf:
            s = preferred_nan_str
        elif self.v == _neginf:
            s = '-' + preferred_inf_str
        elif _isnan(self.v):
            s = preferred_nan_str
        else:
            s = str(self)
        return 'Real(' + repr(s) + ')'

    # arithmetic

    def __neg__(self):
        return Real(-self.v)
        
    def __add__(self, x):
        if isinstance(x, int):
            return Real(self.v + x)
        elif isinstance(x, Real):
            return Real(self.v + x.v)
        else:
            raise ValueError('expected int or Real, got {}'.format(repr(x)))

    def __sub__(self, x):
        if isinstance(x, int):
            return Real(self.v - x)
        elif isinstance(x, Real):
            return Real(self.v - x.v)
        else:
            raise ValueError('expected int or Real, got {}'.format(repr(x)))

    def __mul__(self, x):
        if isinstance(x, int):
            return Real(self.v * x)
        elif isinstance(x, Real):
            return Real(self.v * x.v)
        else:
            raise ValueError('expected int or Real, got {}'.format(repr(x)))

    def __truediv__(self, x):
        if isinstance(x, int):
            return Real(self.v / x)
        elif isinstance(x, Real):
            return Real(self.v / x.v)
        else:
            raise ValueError('expected int or Real, got {}'.format(repr(x)))

    # this might fail right now if you take inf ** -inf, so don't
    def __pow__(self, x):
        if isinstance(x, int):
            return Real(self.v ** x)
        elif isinstance(x, Real):
            return Real(self.v ** x.v)
        else:
            raise ValueError('expected int or Real, got {}'.format(repr(x)))

    # comparison

    def __lt__(self, x):
        if isinstance(x, int):
            return self.v < x
        elif isinstance(x, Real):
            return self.v < x.v
        else:
            raise ValueError('expected int or Real, got {}'.format(repr(x)))

    def __le__(self, x):
        if isinstance(x, int):
            return self.v <= x
        elif isinstance(x, Real):
            return self.v <= x.v
        else:
            raise ValueError('expected int or Real, got {}'.format(repr(x)))

    def __eq__(self, x):
        if isinstance(x, int):
            return self.v == x
        elif isinstance(x, Real):
            return self.v == x.v
        else:
            raise ValueError('expected int or Real, got {}'.format(repr(x)))

    def __ne__(self, x):
        if isinstance(x, int):
            return self.v != x
        elif isinstance(x, Real):
            return self.v != x.v
        else:
            raise ValueError('expected int or Real, got {}'.format(repr(x)))

    def __ge__(self, x):
        if isinstance(x, int):
            return self.v >= x
        elif isinstance(x, Real):
            return self.v >= x.v
        else:
            raise ValueError('expected int or Real, got {}'.format(repr(x)))

    def __gt__(self, x):
        if isinstance(x, int):
            return self.v > x
        elif isinstance(x, Real):
            return self.v > x.v
        else:
            raise ValueError('expected int or Real, got {}'.format(repr(x)))
