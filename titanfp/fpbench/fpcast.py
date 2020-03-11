"""A reusable AST for manipulating or executing FPCores in python."""

import typing


def sexp_to_string(e):
    if isinstance(e, list):
        return '(' + ' '.join((sexp_to_string(x) for x in e)) + ')'
    else:
        return str(e)

def annotation_to_string(e, props, shape=None):
    if props:
        if shape:
            return ('(! ' + ''.join((':' + k + ' ' + sexp_to_string(v) + ' ' for k, v in props.items()))
                    + str(e) + ' '.join((sexp_to_string(dim) for dim in shape)) + ')')
        else:
            return '(! ' + ''.join((':' + k + ' ' + sexp_to_string(v) + ' ' for k, v in props.items())) + str(e) + ')'
    else:
        if shape:
            return '(' + str(e) + ' '.join((sexp_to_string(dim) for dim in shape)) + ')'
        else:
            return str(e)


# base ast class

class Expr(object):
    name: str = 'Expr'


# arbitrary s-expression data (usually from properties)

class Data(Expr):
    name: str = 'Data'

    def __init__(self, value) -> None:
        self.value = value

    def __str__(self):
        return sexp_to_string(self.value)

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.value) + ')'

    def __eq__(self, other):
        if not isinstance(other, Data):
            return False
        return self.value == other.value

    def is_number(self):
        return isinstance(self.value, Val) and not isinstance(self.value, Constant)

    def as_number(self, strict=False):
        if isinstance(self.value, Val) and not isinstance(self.value, Constant):
            return self.value
        elif strict:
            raise TypeError('data is not a number')
        else:
            return None

    def is_symbol(self):
        return isinstance(self.value, Var) or isinstance(self.value, Constant)

    def as_symbol(self, strict=False):
        if isinstance(self.value, Var) or isinstance(self.value, Constant):
            return self.value.value
        elif strict:
            raise TypeError('data is not a symbol')
        else:
            return None

    def is_string(self):
        return isinstance(self.value, String)

    def as_string(self, strict=False):
        if isinstance(self.value, String):
            return self.value.value
        elif strict:
            raise TypeError('data is not a string')
        else:
            return None

    def is_list(self):
        return isinstance(self.value, tuple)

    def as_list(self, strict=False):
        if isinstance(self.value, tuple):
            return self.value
        elif strict:
            raise TypeError('data is not a list')
        else:
            return None


# operations

class NaryExpr(Expr):
    name: str = 'NaryExpr'

    def __init__(self, *children: Expr) -> None:
        self.children: typing.List[Expr] = children

    def __str__(self):
        return '(' + self.name + ''.join((' ' + str(child) for child in self.children)) + ')'

    def __repr__(self):
        return type(self).__name__ + '(' + ', '.join((repr(child) for child in self.children)) + ')'

    def __eq__(self, other):
        if not isinstance(other, NaryExpr):
            return False
        return self.name == other.name and self.children == other.children

class UnknownOperator(NaryExpr):
    name: str = 'UnknownOperator'

    def __init__(self, *children: Expr, name='UnknownOperator') -> None:
        super().__init__(*children)
        self.name = name

    def __repr__(self):
        return type(self).__name__ + '(' + ''.join((repr(child) + ', ' for child in self.children)) + 'name=' + repr(self.name) + ')'

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


# values

class ValueExpr(Expr):
    name: str = 'ValueExpr'

    # Except for integers, all values (variables, constants, or numbers)
    # are represented as strings in the AST.
    def __init__(self, value: str) -> None:
        self.value: str = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.value) + ')'

    def __eq__(self, other):
        if not isinstance(other, ValueExpr):
            return False
        return self.name == other.name and self.value == other.value

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

class Integer(Val):
    name: str = 'Integer'

    def __init__(self, i: int) -> None:
        super().__init__(str(i))
        self.i = i

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.i) + ')'

    def __eq__(self, other):
        if not isinstance(other, Integer):
            return False
        return self.i == other.i

class Rational(Val):
    name: str = 'Rational'

    def __init__(self, p: int, q: int) -> None:
        super().__init__(str(p) + '/' + str(q))
        self.p : int = p
        self.q : int = q

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.p) + ', ' + repr(self.q) + ')'

    def __eq__(self, other):
        if not isinstance(other, Rational):
            return False
        return self.p == other.p and self.q == other.q

class Digits(Val):
    name: str = 'digits'

    def __init__(self, m: int, e: int, b: int) -> None:
        super().__init__(str(m) + ' ' + str(e) + ' ' + str(b))
        self.m: int = m
        self.e: int = e
        self.b: int = b

    def __str__(self):
        return '(' + self.name + ' ' + self.m + ' ' + self.e + ' ' + self.b + ')'

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.m) + ', ' + repr(self.e) + ', ' + repr(self.b) + ')'

    def __eq__(self, other):
        if not isinstance(other, Digits):
            return False
        return self.m == other.m and self.e == other.e and self.b == other.b

class String(ValueExpr):
    name: str = 'String'


# rounding contexts

class Ctx(Expr):
    name: str = '!'

    def __init__(self, props: dict, body: Expr) -> None:
        self.props = props
        self.body = body

    def __str__(self):
        return ('(' + self.name + ' '
                + ''.join((':' + k + ' ' + sexp_to_string(v) + ' ' for k, v in self.props.items()))
                + str(self.body) + ')')

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.props) + ', ' + repr(self.body) + ')'

    def __eq__(self, other):
        if not isinstance(other, Ctx):
            return False
        return self.props == other.props and self.body == other.body


# tensors

class Tensor(Expr):
    name: str = 'tensor'

    def __init__(self, dim_bindings: typing.List[typing.Tuple[str, Expr]], body: Expr) -> None:
        self.dim_bindings: typing.List[typing.Tuple[str, Expr]] = dim_bindings
        self.body: Expr = body

    def __str__(self):
        return ('(' + self.name
                + ' (' + ' '.join(('[' + x + ' ' + str(e) + ']' for x, e in self.dim_bindings)) + ') '
                + str(self.body) + ')')

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.dim_bindings) + ', ' + repr(self.body) + ')'

    def __eq__(self, other):
        if not isinstance(other, Tensor):
            return False
        return self.dim_bindings == other.dim_bindings and self.body == other.body

class TensorStar(Tensor):
    name: str = 'tensor*'

    def __init__(self, ident='', dim_bindings=[], while_bindings=[], body=None) -> None:
        if body is None:
            raise ValueError('must specify a body for TensorStar')
        self.ident: str = ident
        self.dim_bindings: typing.List[typing.Tuple[str, Expr]] = dim_bindings
        self.while_bindings: typing.List[typing.Tuple[str, Expr, Expr]] = while_bindings
        self.body: Expr = body

    def __str__(self):
        if self.ident:
            ident_str = ' ' + self.ident
        else:
            ident_str = ''

        if self.while_bindings:
            while_str = ' (' + ' '.join(('[' + x + ' ' + str(e0) + ' ' + str(e) + ']' for x, e0, e in self.while_bindings)) + ')'
        else:
            while_str = ''
            
        return ('(' + self.name
                + ident_str
                + ' (' + ' '.join(('[' + x + ' ' + str(e) + ']' for x, e in self.dim_bindings)) + ')'
                + while_str
                + ' ' + str(self.body) + ')')

    def __repr__(self):
        return (type(self).__name__ + '('
                + repr(self.ident) + ', '
                + repr(self.dim_bindings) + ', '
                + repr(self.while_bindings) + ', '
                + repr(self.body) + ')')

    def __eq__(self, other):
        if not isinstance(other, ForStar):
            return False
        return (self.ident == other.ident
                and self.dim_bindings == other.dim_bindings
                and self.while_bindings == other.while_bindings
                and self.body == other.body)


# control flow

class If(Expr):
    name: str = 'if'

    def __init__(self, cond: Expr, then_body: Expr, else_body: Expr) -> None:
        self.cond: Expr = cond
        self.then_body: Expr = then_body
        self.else_body: Expr = else_body

    def __str__(self):
        return '(' + self.name + ' ' + str(self.cond) + ' ' + str(self.then_body) + ' ' + str(self.else_body) + ')'

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.cond) + ', ' + repr(self.then_body) + ', ' + repr(self.else_body) + ')'

    def __eq__(self, other):
        if not isinstance(other, If):
            return False
        return self.cond == other.cond and self.else_body == other.else_body and self.then_body == other.then_body

class Let(Expr):
    name: str = 'let'

    def __init__(self, let_bindings: typing.List[typing.Tuple[str, Expr]], body: Expr) -> None:
        self.let_bindings: typing.List[typing.Tuple[str, Expr]] = let_bindings
        self.body: Expr = body

    def __str__(self):
        return ('(' + self.name
                + ' (' + ' '.join(('[' + x + ' ' + str(e) + ']' for x, e in self.let_bindings)) + ') '
                + str(self.body) + ')')

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.let_bindings) + ', ' + repr(self.body) + ')'

    def __eq__(self, other):
        if not isinstance(other, Let):
            return False
        return self.let_bindings == other.let_bindings and self.body == other.body

class LetStar(Let):
    name: str = 'let*'

    def __eq__(self, other):
        if not isinstance(other, LetStar):
            return False
        return self.let_bindings == other.let_bindings and self.body == other.body

class While(Expr):
    name: str = 'while'

    def __init__(self, cond: Expr, while_bindings: typing.List[typing.Tuple[str, Expr, Expr]], body: Expr) -> None:
        self.cond: Expr = cond
        self.while_bindings: typing.List[typing.Tuple[str, Expr, Expr]] = while_bindings
        self.body: Expr = body

    def __str__(self):
        return ('(' + self.name + ' ' + str(self.cond)
                + ' (' + ' '.join(('[' + x + ' ' + str(e0) + ' ' + str(e) + ']' for x, e0, e in self.while_bindings)) + ') '
                + str(self.body) + ')')

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.cond) + ', ' + repr(self.while_bindings) + ', ' + repr(self.body) + ')'

    def __eq__(self, other):
        if not isinstance(other, While):
            return False
        return self.cond == other.cond and self.while_bindings == other.while_bindings and self.body == other.body

class WhileStar(While):
    name: str = 'while*'

    def __eq__(self, other):
        if not isinstance(other, WhileStar):
            return False
        return self.cond == other.cond and self.while_bindings == other.while_bindings and self.body == other.body

class For(Expr):
    name: str = 'for'

    def __init__(self,
                 dim_bindings: typing.List[typing.Tuple[str, Expr]],
                 while_bindings: typing.List[typing.Tuple[str, Expr, Expr]],
                 body: Expr) -> None:
        self.dim_bindings: typing.List[typing.Tuple[str, Expr]] = dim_bindings
        self.while_bindings: typing.List[typing.Tuple[str, Expr, Expr]] = while_bindings
        self.body: Expr = body

    def __str__(self):
        return ('(' + self.name
                + ' (' + ' '.join(('[' + x + ' ' + str(e) + ']' for x, e in self.dim_bindings)) + ')'
                + ' (' + ' '.join(('[' + x + ' ' + str(e0) + ' ' + str(e) + ']' for x, e0, e in self.while_bindings)) + ') '
                + str(self.body) + ')')

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.dim_bindings) + ', ' + repr(self.while_bindings) + ', ' + repr(self.body) + ')'

    def __eq__(self, other):
        if not isinstance(other, For):
            return False
        return self.dim_bindings == other.dim_bindings and self.while_bindings == other.while_bindings and self.body == other.body

class ForStar(For):
    name: str = 'for*'

    def __eq__(self, other):
        if not isinstance(other, ForStar):
            return False
        return self.dim_bindings == other.dim_bindings and self.while_bindings == other.while_bindings and self.body == other.body    


# cast is the identity function, used for repeated rounding

class Cast(UnaryExpr):
    name: str = 'cast'

# tensor operations

class Dim(UnaryExpr):
    name: str = 'dim'

class Size(NaryExpr):
    name: str = 'size'

class Ref(NaryExpr):
    name: str = 'ref'

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
    def __init__(self, inputs, e, props=None, ident=None, name=None, pre=None, spec=None):
        self.inputs = inputs
        self.e = e
        if props is None:
            self.props = {}
        else:
            self.props = props
        self.ident = ident
        self.name = name
        self.pre = pre
        self.spec = spec

    def __str__(self):
        return 'FPCore ({})\n  ident: {}\n  name: {}\n   pre: {}\n  spec: {}\n  {}'.format(
            ' '.join((annotation_to_string(*arg) for arg in self.inputs)),
            str(self.ident), str(self.name), str(self.pre), str(self.spec), str(self.e))

    def __repr__(self):
        return 'FPCore(\n  {},\n  {},\n  ident={}\n  props={}\n)'.format(
            repr(self.inputs), repr(self.e), repr(self.ident), repr(self.props))

    def __eq__(self, other):
        if not isinstance(other, FPCore):
            return False
        return self.inputs == other.inputs and self.e == other.e and self.props == other.props

    @property
    def sexp(self):
        return '(FPCore ({}) {}{})'.format(
            ' '.join((annotation_to_string(*arg) for arg in self.inputs)),
            ''.join(':' + name + ' ' + sexp_to_string(prop) + ' ' for name, prop in self.props.items()),
            str(self.e))
