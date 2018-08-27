"""A reusable AST for manipulating or executing FPCores in python."""


import typing


# base ast classes

class Expr(object):
    name: str = 'Expr'

class NaryExpr(Expr):
    name: str = 'NaryExpr'

    def __init__(self, *children: Expr) -> None:
        self.children: typing.List[Expr] = children

    def __str__(self):
        return '(' + type(self).name + ''.join((' ' + str(child) for child in self.children)) + ')'

    def __repr__(self):
        return type(self).__name__ + '(' + ', '.join((repr(child) for child in self.children)) + ')'

class UnaryExpr(NaryExpr):
    name: str = 'UnaryExpr'

    def __init__(self, child0: Expr) -> None:
        super().__init__(child0)

class BinaryExpr(NaryExpr):
    name: str = 'BinaryExpr'

    def __init__(self, child0: Expr, child1: Expr) -> None:
        super().__init__(child0, child1)

class TernaryExpr(NaryExpr):
    name: str = 'TernaryExpr'

    def __init__(self, child0: Expr, child1: Expr, child2: Expr) -> None:
        super().__init__(child0, child1, child2)

class ValueExpr(Expr):
    name: str = 'ValueExpr'

    # All values (variables, constants, or numbers) are represented as strings in the AST.
    def __init__(self, value: str) -> None:
        self.value: str = value

    def __str__(self):
        return self.value

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.value) + ')'


# rounding contexts

class Ctx(Expr):
    name: str = '!'

    def __init__(self, props: dict, body: Expr) -> None:
        self.props = props
        self.body = body

    def __str__(self):
        return ('(' + type(self).name + ' '
                + ''.join((':' + k + ' ' + str(v) + ' ' for k, v in self.props.items()))
                + str(self.body) + ')')

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.props) + ', ' + repr(self.body) + ')'


# values

class Var(ValueExpr):
    name: str = 'Var'

class Val(ValueExpr):
    name: str = 'Val'

class Constant(Val):
    name: str = 'Constant'

class Decnum(Val):
    name: str = 'Decnum'

class Hexnum(Val):
    name: str = 'Hexnum'

class Rational(Val):
    name: str = 'Rational'

    def __init__(self, p: int, q: int) -> None:
        super().__init__(str(p) + '/' + str(q))
        self.p : int = p
        self.q : int = q

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.p) + ', ' + repr(self.q) + ')'

class Digits(Val):
    name: str = 'digits'

    def __init__(self, m: int, e: int, b: int) -> None:
        super().__init__(str(m) + ' ' + str(e) + ' ' + str(b))
        self.m: int = m
        self.e: int = e
        self.b: int = b

    def __str__(self):
        return '(' + type(self).name + ' ' + self.m + ' ' + self.e + ' ' + self.b + ')'

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.m) + ', ' + repr(self.e) + ', ' + repr(self.b) + ')'


# control flow

class If(Expr):
    name: str = 'if'

    def __init__(self, cond: Expr, then_body: Expr, else_body: Expr) -> None:
        self.cond: Expr = cond
        self.then_body: Expr = then_body
        self.else_body: Expr = else_body

    def __str__(self):
        return '(' + type(self).name + ' ' + str(self.cond) + ' ' + str(self.then_body) + ' ' + str(self.else_body) + ')'

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.cond) + ', ' + repr(self.then_body) + ', ' + repr(self.else_body) + ')'

class Let(Expr):
    name: str = 'let'

    def __init__(self, let_bindings: typing.List[typing.Tuple[str, Expr]], body: Expr) -> None:
        self.let_bindings: typing.List[typing.Tuple[str, Expr]] = let_bindings
        self.body: Expr = body

    def __str__(self):
        return ('(' + type(self).name
                + ' (' + ' '.join(('[' + x + ' ' + str(e) + ']' for x, e in self.let_bindings)) + ') '
                + str(self.body) + ')')

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.let_bindings) + ', ' + repr(self.body) + ')'

class While(Expr):
    name: str = 'while'

    def __init__(self, cond: Expr, while_bindings: typing.List[typing.Tuple[str, Expr, Expr]], body: Expr) -> None:
        self.cond: Expr = cond
        self.while_bindings: typing.List[typing.Tuple[str, Expr, Expr]] = while_bindings
        self.body: Expr = body

    def __str__(self):
        return ('(' + type(self).name + ' ' + str(self.cond)
                + ' (' + ' '.join(('[' + x + ' ' + str(e0) + ' ' + str(e) + ']' for x, e0, e in self.while_bindings)) + ') '
                + str(self.body) + ')')

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.cond) + ', ' + repr(self.while_bindings) + ', ' + repr(self.body) + ')'


# cast is the identity function, used for repeated rounding

class Cast(UnaryExpr):
    name: str = 'cast'


# IEEE 754 required arithmetic

class Add(BinaryExpr):
    name: str = '+'

class Sub(BinaryExpr):
    name: str = '-'

class Mul(BinaryExpr):
    name: str = '*'

class Div(BinaryExpr):
    name: str = '/'

class Sqrt(UnaryExpr):
    name: str = 'sqrt'

class Fma(TernaryExpr):
    name: str = 'fma'

# discrete operations

class Neg(UnaryExpr):
    # note that unary negation has the same "name" as subtraction
    name: str = '-'

class Copysign(BinaryExpr):
    name: str = 'copysign'

class Fabs(UnaryExpr):
    name: str = 'fabs'

# composite arithmetic

class Fdim(BinaryExpr):
    name: str = 'fdim'

class Fmax(BinaryExpr):
    name: str = 'fmax'

class Fmin(BinaryExpr):
    name: str = 'fmin'

class Fmod(BinaryExpr):
    name: str = 'fmod'

class Remainder(BinaryExpr):
    name: str = 'remainder'

# rounding and truncation

class Ceil(UnaryExpr):
    name: str = 'ceil'

class Floor(UnaryExpr):
    name: str = 'floor'

class Nearbyint(UnaryExpr):
    name: str = 'nearbyint'

class Round(UnaryExpr):
    name: str = 'round'

class Trunc(UnaryExpr):
    name: str = 'trunc'

# trig

class Acos(UnaryExpr):
    name: str = 'acos'

class Acosh(UnaryExpr):
    name: str = 'acosh'

class Asin(UnaryExpr):
    name: str = 'asin'

class Asinh(UnaryExpr):
    name: str = 'asinh'

class Atan(UnaryExpr):
    name: str = 'atan'

class Atan2(BinaryExpr):
    name: str = 'atan2'

class Atanh(UnaryExpr):
    name: str = 'atanh'

class Cos(UnaryExpr):
    name: str = 'cos'

class Cosh(UnaryExpr):
    name: str = 'cosh'

class Sin(UnaryExpr):
    name: str = 'sin'

class Sinh(UnaryExpr):
    name: str = 'sinh'

class Tan(UnaryExpr):
    name: str = 'tan'

class Tanh(UnaryExpr):
    name: str = 'tanh'

# exponentials

class Exp(UnaryExpr):
    name: str = 'exp'

class Exp2(UnaryExpr):
    name: str = 'exp2'

class Expm1(UnaryExpr):
    name: str = 'expm1'

class Log(UnaryExpr):
    name: str = 'log'

class Log10(UnaryExpr):
    name: str = 'log10'

class Log1p(UnaryExpr):
    name: str = 'log1p'

class Log2(UnaryExpr):
    name: str = 'log2'

# powers

class Cbrt(UnaryExpr):
    name: str = 'cbrt'

class Hypot(BinaryExpr):
    name: str = 'hypot'

class Pow(BinaryExpr):
    name: str = 'pow'

# other

class Erf(UnaryExpr):
    name: str = 'erf'

class Erfc(UnaryExpr):
    name: str = 'erfc'

class Lgamma(UnaryExpr):
    name: str = 'lgamma'

class Tgamma(UnaryExpr):
    name: str = 'tgamma'


# comparison

class LT(NaryExpr):
    name: str = '<'

class GT(NaryExpr):
    name: str = '>'

class LEQ(NaryExpr):
    name: str = '<='

class GEQ(NaryExpr):
    name: str = '>='

class EQ(NaryExpr):
    name: str = '=='

class NEQ(NaryExpr):
    name: str = '!='

# classification

class Isfinite(UnaryExpr):
    name: str = 'isfinite'

class Isinf(UnaryExpr):
    name: str = 'isinf'

class Isnan(UnaryExpr):
    name: str = 'isnan'

class Isnormal(UnaryExpr):
    name: str = 'isnormal'

class Signbit(UnaryExpr):
    name: str = 'signbit'

# logic

class And(NaryExpr):
    name: str = 'and'

class Or(NaryExpr):
    name: str = 'or'

class Not(UnaryExpr):
    name: str = 'not'


# fpcore objects and helpers

class FPCore(object):
    def __init__(self, inputs, e, props = None):
        self.inputs = inputs
        self.e = e
        if props is None:
            self.props = {}
        else:
            self.props = props

        self.name = self.props.get('name', None)
        self.pre = self.props.get('pre', None)
        self.spec = self.props.get('spec', None)

    def __str__(self):
        return 'FPCore ({})\n  name: {}\n   pre: {}\n  spec: {}\n  {}'.format(
            ' '.join((_annotate_input(name, props) for name, props in self.inputs)),
            self.name, self.pre, self.spec, self.e)

    def __repr__(self):
        return 'FPCore(\n  {},\n  {},\n  props={}\n)'.format(
            repr(self.inputs), repr(self.e), repr(self.props))

    @property
    def sexp(self):
        return '(FPCore ({}) {}{})'.format(
            ' '.join((_annotate_input(name, props) for name, props in self.inputs)),
            ''.join(':' + name + ' ' + _prop_to_sexp(prop) + ' ' for name, prop in self.props.items()),
            str(self.e))

def _annotate_input(name, props):
    if props:
        return '(! ' + ''.join((':' + k + ' ' + str(v) + ' ' for k, v in props.items())) + name + ')'
    else:
        return name

def _prop_to_sexp(p):
    if isinstance(p, str):
        return '"' + p + '"'
    elif isinstance(p, list):
        return '(' + ' '.join(p) + ')'
    else:
        return str(p)
