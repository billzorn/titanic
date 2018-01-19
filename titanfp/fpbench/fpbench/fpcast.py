import operator
import math
import typing

# on a side note, it looks lists are always faster than tuples, at least in anaconda cpython

# pairwise implementations of n-ary comparisons; these don't short-circuit

def _impl_by_pairs(op, conj):
    def impl(args_iter):
        args = [*args_iter]
        if len(args) <= 2:
            return op(*args)
        else:
            return conj(op(args[i], args[i+1]) for i in range(len(args)-1))
    return impl

def _impl_all_pairs(op, conj):
    def impl(args_iter):
        args = [*args_iter]
        if len(args) <= 2:
            return op(*args)
        else:
            return conj(op(args[i], args[j]) for i in range(len(args)-1) for j in range(i+1,len(args)))
    return impl

# evaluation context

class EvalCtx(object):

    def __init__(self, inputs):
        self.inputs = inputs
        self.variables = inputs

    def let(self, bindings):
        self.variables = self.variables.update(bindings)
        return self

# ast

class Expr(object):
    name: str = 'Expr'

    def __init__(self, *children: "Expr") -> None:
        self.children: typing.Tuple[Expr, ...] = children

    def __str__(self):
        return '(' + type(self).name + ''.join((' ' + str(child) for child in self.children)) + ')'

    def __repr__(self):
        return type(self).__name__ + '(' + ', '.join((repr(child) for child in self.children)) + ')'

    def __call__(self, *args, **kwargs):
        return self.evaluate(EvalCtx(*args, **kwargs))

    def evaluate(self, ctx: EvalCtx):
        raise ValueError('unimplemented ast node ' + type(self).__name__)

class UnaryExpr(Expr):
    name: str = 'UnaryExpr'

    def __init__(self, child0: Expr) -> None:
        self.children: typing.Tuple[Expr, ...] = (child0,)

class BinaryExpr(Expr):
    name: str = 'BinaryExpr'

    def __init__(self, child0: Expr, child1: Expr) -> None:
        self.children: typing.Tuple[Expr, ...] = (child0, child1)

class ValueExpr(Expr):
    name: str = 'ValueExpr'

    # All values (variables, constants, or numbers) are represented as strings
    # in the AST.
    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self):
        return self.value

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.value) + ')'

# values

class Val(ValueExpr):
    name: str = 'Val'

    def evaluate(self, ctx: EvalCtx):
        return float(self.value) # TODO

class Var(ValueExpr):
    name: str = 'Var'

    def evaluate(self, ctx: EvalCtx):
        return float(ctx.variables[self.value]) # TODO

# control flow

class If(Expr):
    name: str = 'if'

    def __init__(self, cond: Expr, then_body: Expr, else_body: Expr) -> None:
        self.children: typing.Tuple[Expr, ...] = (cond, then_body, else_body)

    def evaluate(self, ctx: EvalCtx):
        cond, then_body, else_body = self.children
        if cond.evaluate(ctx):
            return then_body.evaluate(ctx)
        else:
            return else_body.evaluate(ctx)

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

    def evaluate(self, ctx: EvalCtx):
        return self.body.evaluate(ctx.let({x: e.evaluate(ctx) for x, e in self.let_bindings}))

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

    def evaluate(self, ctx: EvalCtx):
        bindings = {x: e0 for x, e0, _ in self.while_bindings}
        while self.cond.evaluate(ctx.let(bindings)):
            bindings = {x: e.evaluate(ctx.let(bindings)) for x, _, e in self.while_bindings}
        return self.body.evaluate(ctx.let(bindings))

# arithmetic

class Neg(UnaryExpr):
    name: str = 'neg'

    def __str__(self):
        return '(-' + ''.join((' ' + str(child) for child in self.children)) + ')'

    def evaluate(self, ctx: EvalCtx):
        (child,) = self.children
        return -child.evaluate(ctx)

class Sqrt(UnaryExpr):
    name: str = 'sqrt'

    def evaluate(self, ctx: EvalCtx):
        (child,) = self.children
        return math.sqrt(child.evaluate(ctx))

class Add(BinaryExpr):
    name: str = '+'

    def evaluate(self, ctx: EvalCtx):
        left, right = self.children
        return left.evaluate(ctx) + right.evaluate(ctx)

class Sub(BinaryExpr):
    name: str = '-'

    def evaluate(self, ctx: EvalCtx):
        left, right = self.children
        return left.evaluate(ctx) - right.evaluate(ctx)

class Mul(BinaryExpr):
    name: str = '*'

    def evaluate(self, ctx: EvalCtx):
        left, right = self.children
        return left.evaluate(ctx) * right.evaluate(ctx)

class Div(BinaryExpr):
    name: str = '/'

    def evaluate(self, ctx: EvalCtx):
        left, right = self.children
        return left.evaluate(ctx) / right.evaluate(ctx)

# comparison

_nary_lt = _impl_by_pairs(operator.lt, all)
_nary_gt = _impl_by_pairs(operator.gt, all)
_nary_le = _impl_by_pairs(operator.le, all)
_nary_ge = _impl_by_pairs(operator.ge, all)
_nary_eq = _impl_by_pairs(operator.eq, all)
_nary_ne = _impl_all_pairs(operator.ne, all)


class LT(Expr):
    name: str = '<'

    def evaluate(self, ctx: EvalCtx):
        return _nary_lt(child.evaluate(ctx) for child in self.children)

class GT(Expr):
    name: str = '>'

    def evaluate(self, ctx: EvalCtx):
        return _nary_gt(child.evaluate(ctx) for child in self.children)

class LEQ(Expr):
    name: str = '<='

    def evaluate(self, ctx: EvalCtx):
        return _nary_le(child.evaluate(ctx) for child in self.children)

class GEQ(Expr):
    name: str = '>='

    def evaluate(self, ctx: EvalCtx):
        return _nary_ge(child.evaluate(ctx) for child in self.children)

class EQ(Expr):
    name: str = '=='

    def evaluate(self, ctx: EvalCtx):
        return _nary_eq(child.evaluate(ctx) for child in self.children)

class NEQ(Expr):
    name: str = '!='

    def evaluate(self, ctx: EvalCtx):
        return _nary_ne(child.evaluate(ctx) for child in self.children)

# logic

class And(Expr):
    name: str = 'and'

    def evaluate(self, ctx: EvalCtx):
        return all(*[child.evaluate(ctx) for child in self.children])

class Or(Expr):
    name: str = 'or'

    def evaluate(self, ctx: EvalCtx):
        return any(*[child.evaluate(ctx) for child in self.children])

class Not(UnaryExpr):
    name: str = 'not'

    def evaluate(self, ctx: EvalCtx):
        (child,) = self.children
        return not child.evaluate(ctx)
