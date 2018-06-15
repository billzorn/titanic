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

class Val(ValueExpr):
    name: str = 'Val'

class Var(ValueExpr):
    name: str = 'Var'

class Digits(Expr):
    name: str = 'digits'

    def __init__(self, m: str, e: str, b: str) -> None:
        self.m: str = m
        self.e: int = int(e)
        self.b: int = int(b)

    def __str__(self):
        return '(' + type(self).name + ' ' + self.m + ' ' + str(self.e) + ' ' + str(self.b) + ')'

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.m) + ', ' + repr(str(self.e)) + ', ' + repr(str(self.b)) + ')'


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


# arithmetic

class Neg(UnaryExpr):
    # note that unary negation has the same "name" as subtraction
    name: str = '-'

class Sqrt(UnaryExpr):
    name: str = 'sqrt'

class Add(BinaryExpr):
    name: str = '+'

class Sub(BinaryExpr):
    name: str = '-'

class Mul(BinaryExpr):
    name: str = '*'

class Div(BinaryExpr):
    name: str = '/'

# more arithmetic

class Floor(UnaryExpr):
    name: str = 'floor'

class Fmod(BinaryExpr):
    name: str = 'fmod'

class Pow(BinaryExpr):
    name: str = 'pow'

class Sin(UnaryExpr):
    name: str = 'sin'

class Acos(UnaryExpr):
    name: str = 'acos'


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
        return '(FPCore ({}) {} {})'.format(
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

def _canonicalize_expr(e, props, whilelist={'precision', 'round'}, blacklist=set()):
    raise ValueError('unimplemented')
