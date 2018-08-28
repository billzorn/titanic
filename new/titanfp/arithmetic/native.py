"""FPCore interpreter using Python's math module."""


import math
# import fractions

from . import interpreter


class Interpreter(interperter.SimpleInterpreter):

    # datatype conversion

    dtype = float
    ctype = EvalCtx

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

    # disabled for now
    # @classmethod
    # def _eval_fma(cls, e, ctx):
    #     raise ValueError('unimplemented: fma')

    #     child0 = cls.evaluate(e.children[0], ctx)
    #     child1 = cls.evaluate(e.children[1], ctx)
    #     child2 = cls.evaluate(e.children[2], ctx)

    #     # thanks to Python issue 29282
    #     # https://bugs.python.org/issue29282

    #     if math.isnan(child0):
    #         return child0
    #     elif math.isnan(child1):
    #         return child1
    #     # Intel style: inf * 0 + nan returns the nan
    #     elif math.isnan(child2):
    #         return child2
    #     elif (math.isinf(child0) and child1 == 0.0) or (child0 == 0.0 and math.isinf(child1)):
    #         return math.nan

    #     # get the signs
    #     sign_01 = math.copysign(1.0, child0) * math.copysign(1.0, child1)
    #     sign_2 = math.copysign(1.0, child2)

    #     # other nasty cases
    #     if math.isinf(child0) or math.isinf(child1):
    #         if math.isinf(child2) and sign_01 != sign_2:
    #             return math.nan
    #         else:
    #             return math.inf * sign_01
    #     elif math.isinf(child2):
    #         return child2

    #     # compute result with Fractions
    #     result = (fractions.Fraction(child0) * fractions.Fraction(child1)) + fractions.Fraction(child2)

    #     # fix up sign of zero
    #     if result == 0:
    #         if sign_01 == sign_2 == -1.0:
    #             return -0.0
    #         else:
    #             return 0.0
    #     else:
    #         try:
    #             f = float(result)
    #         except OverflowError:
    #             if result > 0:
    #                 f = math.inf
    #             else:
    #                 f = -math.inf
    #         return f

    @classmethod
    def _eval_copysign(cls, e, ctx):
        return math.copysign(cls.evaluate(e.children[0], ctx), cls.evaluate(e.children[1], ctx))

    @classmethod
    def _eval_fabs(cls, e, ctx):
        return math.fabs(cls.evaluate(e.children[0], ctx))
