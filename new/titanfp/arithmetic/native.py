"""FPCore interpreter using Python's math module."""


import math
# import fractions

from . import interpreter
from . import evalctx


_SMALLEST_NORMAL = 2 ** -1022


class Interpreter(interpreter.SimpleInterpreter):
    """FPCore interpreter using builtin Python floats.
    Most operations provided by the math modules; some emulated.
    """

    # datatype conversion

    dtype = float
    ctype = evalctx.EvalCtx

    constants = {
        'TRUE': True,
        'FALSE': False,
        'PI': math.pi,
        'E': math.e,
    }

    @staticmethod
    def arg_to_digital(x, ctx):
        return float(x)

    @staticmethod
    def round_to_context(x, ctx):
        # do nothing
        return x


    # values

    @classmethod
    def _eval_constant(cls, e, ctx):
        try:
            return cls.constants[e.value]
        except KeyError as exn:
            raise ValueError('unsupported constant {}'.format(repr(exn.args[0])))

    @classmethod
    def _eval_decnum(cls, e, ctx):
        return cls.arg_to_digital(e.value)

    @classmethod
    def _eval_hexnum(cls, e, ctx):
        return float.fromhex(e.value)

    def _eval_rational(cls, e, ctx):
        try:
            return e.p / e.q
        except OverflowError:
            return math.inf * math.copysign(1.0, e.p)

    @classmethod
    def _eval_digits(cls, e, ctx):
        digits = compute_digits(e.m, e.e, e.b, prec=53)
        # TODO: not guaranteed correct rounding, return code is ignored!
        return float(gmpmath.digital_to_mpfr(digits))


    # arithmetic

    @classmethod
    def _eval_div(cls, e, ctx):
        child0 = cls.evaluate(e.children[0], ctx)
        child1 = cls.evaluate(e.children[1], ctx)
        if child1 == 0.0:
            if child0 == 0.0:
                return math.nan * math.copysign(1.0, child0)
            else:
                return math.inf * math.copysign(1.0, child0)
        else:
            return child0 / child1

    @classmethod
    def _eval_sqrt(cls, e, ctx):
        return math.sqrt(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_fma(cls, e, ctx):
        child0 = cls.evaluate(e.children[0], ctx)
        child1 = cls.evaluate(e.children[1], ctx)
        child2 = cls.evaluate(e.children[2], ctx)

        # thanks to Python issue 29282
        # https://bugs.python.org/issue29282

        if math.isnan(child0):
            return child0
        elif math.isnan(child1):
            return child1
        # Intel style: inf * 0 + nan returns the nan
        elif math.isnan(child2):
            return child2
        elif (math.isinf(child0) and child1 == 0.0) or (child0 == 0.0 and math.isinf(child1)):
            return math.nan

        # get the signs
        sign_01 = math.copysign(1.0, child0) * math.copysign(1.0, child1)
        sign_2 = math.copysign(1.0, child2)

        # other nasty cases
        if math.isinf(child0) or math.isinf(child1):
            if math.isinf(child2) and sign_01 != sign_2:
                return math.nan
            else:
                return math.inf * sign_01
        elif math.isinf(child2):
            return child2

        # compute result with Fractions
        result = (fractions.Fraction(child0) * fractions.Fraction(child1)) + fractions.Fraction(child2)

        # fix up sign of zero
        if result == 0:
            if sign_01 == sign_2 == -1.0:
                return -0.0
            else:
                return +0.0
        else:
            try:
                f = float(result)
            except OverflowError:
                if result > 0:
                    f = math.inf
                else:
                    f = -math.inf
            return f

    @classmethod
    def _eval_copysign(cls, e, ctx):
        return math.copysign(cls.evaluate(e.children[0], ctx), cls.evaluate(e.children[1], ctx))

    @classmethod
    def _eval_fdim(cls, e, ctx):
        child0 = cls.evaluate(e.children[0], ctx)
        child1 = cls.evaluate(e.children[1], ctx)
        if child0 > child1:
            return child0 - child1
        else:
            return +0.0

    @classmethod
    def _eval_fmax(cls, e, ctx):
        child0 = cls.evaluate(e.children[0], ctx)
        child1 = cls.evaluate(e.children[1], ctx)
        if math.isnan(child0):
            return child1
        elif math.isnan(child1):
            return child0
        else:
            return max(child0, child1)

    @classmethod
    def _eval_fmin(cls, e, ctx):
        child0 = cls.evaluate(e.children[0], ctx)
        child1 = cls.evaluate(e.children[1], ctx)
        if math.isnan(child0):
            return child1
        elif math.isnan(child1):
            return child0
        else:
            return min(child0, child1)

    @classmethod
    def _eval_fmod(cls, e, ctx):
        return math.fmod(cls.evaluate(e.children[0], ctx), cls.evaluate(e.children[1], ctx))

    @classmethod
    def _eval_remainder(cls, e, ctx):
        raise ValueError('remainder: no native implementation, emulation unsupported')

    @classmethod
    def _eval_ceil(cls, e, ctx):
        f, i = math.modf(cls.evaluate(e.children[0], ctx))
        if f > 0.0:
            return i + 1.0
        else:
            return i

    @classmethod
    def _eval_floor(cls, e, ctx):
        f, i = math.modf(cls.evaluate(e.children[0], ctx))
        if f < 0.0:
            return i - 1.0
        else:
            return i

    @classmethod
    def _eval_nearbyint(cls, e, ctx):
        f, i = math.modf(cls.evaluate(e.children[0], ctx))
        if abs(f) > 0.5 or (abs(f) == 0.5 and int(i) % 2 == 1):
            return i + math.copysign(1.0, f)
        else:
            return i

    @classmethod
    def _eval_round(cls, e, ctx):
        f, i = math.modf(cls.evaluate(e.children[0], ctx))
        if abs(f) >= 0.5:
            return i + math.copysign(1.0, f)
        else:
            return i

    @classmethod
    def _eval_trunc(cls, e, ctx):
        f, i = math.modf(cls.evaluate(e.children[0], ctx))
        return i

    @classmethod
    def _eval_acos(cls, e, ctx):
        return math.acos(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_acosh(cls, e, ctx):
        return math.acosh(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_asin(cls, e, ctx):
        return math.asin(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_asinh(cls, e, ctx):
        return math.asinh(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_atan(cls, e, ctx):
        return math.atan(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_atan2(cls, e, ctx):
        return math.atan2(cls.evaluate(e.children[0], ctx), cls.evaluate(e.children[1], ctx))

    @classmethod
    def _eval_atanh(cls, e, ctx):
        return math.atanh(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_cos(cls, e, ctx):
        return math.cos(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_cosh(cls, e, ctx):
        return math.cosh(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_sin(cls, e, ctx):
        return math.sinh(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_sinh(cls, e, ctx):
        return math.sinh(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_tan(cls, e, ctx):
        return math.tan(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_tanh(cls, e, ctx):
        return math.tanh(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_exp(cls, e, ctx):
        return math.exp(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_exp2(cls, e, ctx):
        return 2.0 ** cls.evaluate(e.children[0], ctx)

    @classmethod
    def _eval_expm1(cls, e, ctx):
        return math.expm1(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_log(cls, e, ctx):
        return math.log(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_log10(cls, e, ctx):
        return math.log10(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_log1p(cls, e, ctx):
        return math.log1p(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_log2(cls, e, ctx):
        return math.log2(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_cbrt(cls, e, ctx):
        raise ValueError('cbrt: no native implementation, emulation unsupported')

    @classmethod
    def _eval_hypot(cls, e, ctx):
        return math.hypot(cls.evaluate(e.children[0], ctx), cls.evaluate(e.children[1], ctx))

    @classmethod
    def _eval_pow(cls, e, ctx):
        return cls.evaluate(e.children[0], ctx) ** cls.evaluate(e.children[1], ctx)

    @classmethod
    def _eval_erf(cls, e, ctx):
        return math.erf(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_erfc(cls, e, ctx):
        return math.erfc(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_log(cls, e, ctx):
        return math.log(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_lgamma(cls, e, ctx):
        return math.lgamma(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_tgamma(cls, e, ctx):
        return math.gamma(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_isfinite(cls, e, ctx):
        return math.isfinite(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_isinf(cls, e, ctx):
        return math.isinf(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_isnan(cls, e, ctx):
        return math.isnan(cls.evaluate(e.children[0], ctx))

    @classmethod
    def _eval_isnormal(cls, e, ctx):
        child0 = cls.evaluate(e.children[0], ctx)
        return ((not math.isnan(child0)) and
                (not math.isinf(child0)) and
                (not abs(child0) < _SMALLEST_NORMAL))

    @classmethod
    def _eval_signbit(cls, e, ctx):
        return math.copysign(1.0, cls.evaluate(e.children[0], ctx)) < 0.0

    @classmethod
    def evaluate(cls, e, ctx):
        try:
            return super().evaluate(e, ctx)
        except ValueError as exn:
            if len(exn.args) == 1 and exn.args[0].strip().lower() == 'math domain error':
                return math.nan
            else:
                raise exn
