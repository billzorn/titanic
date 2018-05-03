"""A reusable AST for manipulating or executing FPCores in python."""


import typing


# base ast classes

class Expr(object):
    name: str = 'Expr'
    props = {}

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
        self.children: typing.List[Expr] = [child0,]

class BinaryExpr(NaryExpr):
    name: str = 'BinaryExpr'

    def __init__(self, child0: Expr, child1: Expr) -> None:
        self.children: typing.List[Expr] = [child0, child1]

class ValueExpr(Expr):
    name: str = 'ValueExpr'

    # All values (variables, constants, or numbers) are represented as strings
    # in the AST.
    def __init__(self, value: str) -> None:
        self.value: str = value

    def __str__(self):
        return self.value

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.value) + ')'


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


# control flow

class If(NaryExpr):
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

class Fmod(BinaryExpr):
    name: str = 'fmod'

class Floor(UnaryExpr):
    name: str = 'floor'

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
