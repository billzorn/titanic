import operator
import math
import typing

import z3


# hacked in math code, it's k

import numpy

# on a side note, it looks lists are always faster than tuples, at least in anaconda cpython

# temporary haxxxx
_Z3SORT = z3.FPSort(5, 11)
_Z3RM = z3.RoundNearestTiesToEven()


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

# evaluation contexts

class EvalCtx(object):

    def __init__(self, inputs):
        self.inputs = inputs
        self.variables = inputs

    def let(self, bindings):
        self.variables = self.variables.update(bindings)
        return self

class SigCtx(object):
    def __init__(self, x, lsb=None, lsb_min=None):
        self.x = float(x) # TODO: raw python floats

        m, e = math.frexp(self.x)
        if lsb is None:
            self.lsb = e - 53
        else:
            self.lsb = lsb
        if lsb_min is None:
            self.lsb_min = self.lsb
        else:
            self.lsb_min = lsb_min

        #print(repr(self))

    def __repr__(self):
        m, e = math.frexp(self.x)
        return ("x: {}\n  lsb:     {}\n  lsb_min: {}\n  precision:     {}\n  precision lost: {}"
                .format(repr(self.x), repr(self.lsb), repr(self.lsb_min), repr(e - self.lsb), repr(self.lsb - self.lsb_min)))

# ast

class Expr(object):
    name: str = 'Expr'

    def __init__(self):
        self._z3_expr = None
        self._sigctx = None

    @property
    def z3_expr(self):
        if self._z3_expr is None:
            self._z3_expr = self._translate_z3()
        return self._z3_expr

    def __call__(self, *args, **kwargs):
        return self.evaluate(EvalCtx(*args, **kwargs))

    def evaluate(self, ctx: EvalCtx):
        raise ValueError('evaluation unimplemented for ast node ' + type(self).__name__)

    def evaluate_sig(self, ctx: EvalCtx):
        raise ValueError('significance unimplemented for ast node ' + type(self).__name__)

    def _translate_z3(self):
        raise ValueError('z3 compilation not implemented for ast node ' + type(self).__name__)

class NaryExpr(Expr):
    name: str = 'NaryExpr'

    def __init__(self, *children: "Expr") -> None:
        super().__init__()
        self.children: typing.List[Expr] = children

    def __str__(self):
        return '(' + type(self).name + ''.join((' ' + str(child) for child in self.children)) + ')'

    def __repr__(self):
        return type(self).__name__ + '(' + ', '.join((repr(child) for child in self.children)) + ')'


class UnaryExpr(NaryExpr):
    name: str = 'UnaryExpr'

    def __init__(self, child0: Expr) -> None:
        super().__init__()
        self.children: typing.Luple[Expr] = [child0,]

class BinaryExpr(NaryExpr):
    name: str = 'BinaryExpr'

    def __init__(self, child0: Expr, child1: Expr) -> None:
        super().__init__()
        self.children: typing.List[Expr] = [child0, child1]


class ValueExpr(Expr):
    name: str = 'ValueExpr'

    # All values (variables, constants, or numbers) are represented as strings
    # in the AST.
    def __init__(self, value: str) -> None:
        super().__init__()
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

    def evaluate_sig(self, ctx: EvalCtx):
        return SigCtx(self.value)

    def _translate_z3(self):
        return z3.FPVal(self.value, _Z3SORT) # TODO

class Var(ValueExpr):
    name: str = 'Var'

    def evaluate(self, ctx: EvalCtx):
        return float(ctx.variables[self.value]) # TODO

    def evaluate_sig(self, ctx: EvalCtx):
        return SigCtx(ctx.variables[self.value])

    def _translate_z3(self):
        return z3.FP(self.value, _Z3SORT) # TODO

# control flow

class If(NaryExpr):
    name: str = 'if'

    def __init__(self, cond: Expr, then_body: Expr, else_body: Expr) -> None:
        super().__init__()
        self.children: typing.List[Expr] = [cond, then_body, else_body]

    def evaluate(self, ctx: EvalCtx):
        cond, then_body, else_body = self.children
        if cond.evaluate(ctx):
            return then_body.evaluate(ctx)
        else:
            return else_body.evaluate(ctx)

    def evaluate_sig(self, ctx: EvalCtx):
        cond, then_body, else_body = self.children
        if cond.evaluate(ctx): # TODO - uses raw evaluate without significance tracking
            return then_body.evaluate_sig(ctx)
        else:
            return else_body.evaluate_sig(ctx)

    def _translate_z3(self):
        cond, then_body, else_body = self.children
        return z3.If(cond.z3_expr, then_body.z3_expr, else_body.z3_expr)

class Let(Expr):
    name: str = 'let'

    def __init__(self, let_bindings: typing.List[typing.Tuple[str, Expr]], body: Expr) -> None:
        super().__init__()
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
        super.__init__()
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
    # note that unary negation has the same "name" as subtraction
    name: str = '-'

    def evaluate(self, ctx: EvalCtx):
        (child,) = self.children
        return -child.evaluate(ctx)

    def evaluate_sig(self, ctx: EvalCtx):
        (child,) = self.children
        sctx = child.evaluate_sig(ctx)
        return SigCtx(-sctx.x, sctx.lsb, sctx.lsb_min)

    def _translate_z3(self):
        (child,) = self.children
        return z3.fpNeg(_Z3RM, child.z3_expr)


class Sqrt(UnaryExpr):
    name: str = 'sqrt'

    def evaluate(self, ctx: EvalCtx):
        (child,) = self.children
        return math.sqrt(child.evaluate(ctx))

    def evaluate_sig(self, ctx: EvalCtx):
        (child,) = self.children
        child_sig = child.evaluate_sig(ctx)
        child_e, child_m = math.frexp(child_sig.x)
        
        sigbits = child_e - child_sig.lsb
        maxbits = child_e - child_sig.lsb_min
        x = math.sqrt(child_sig.x)
        m, e = math.frexp(x)

        return SigCtx(x, e - sigbits, e - maxbits)

    def _translate_z3(self):
        (child,) = self.children
        return z3.fpSqrt(_Z3RM, child.z3_expr)

class Add(BinaryExpr):
    name: str = '+'

    def evaluate(self, ctx: EvalCtx):
        left, right = self.children
        return left.evaluate(ctx) + right.evaluate(ctx)

    def evaluate_sig(self, ctx: EvalCtx):
        left, right = self.children
        left_sig = left.evaluate_sig(ctx)
        right_sig = right.evaluate_sig(ctx)
        return SigCtx(left_sig.x + right_sig.x, max(left_sig.lsb, right_sig.lsb), min(left_sig.lsb_min, right_sig.lsb_min))

    def _translate_z3(self):
        left, right = self.children
        return z3.fpAdd(_Z3RM, left.z3_expr, right.z3_expr)

class Sub(BinaryExpr):
    name: str = '-'

    def evaluate(self, ctx: EvalCtx):
        left, right = self.children
        return left.evaluate(ctx) - right.evaluate(ctx)

    def evaluate_sig(self, ctx: EvalCtx):
        left, right = self.children
        left_sig = left.evaluate_sig(ctx)
        right_sig = right.evaluate_sig(ctx)
        return SigCtx(left_sig.x - right_sig.x, max(left_sig.lsb, right_sig.lsb), min(left_sig.lsb_min, right_sig.lsb_min))

    def _translate_z3(self):
        left, right = self.children
        return z3.fpSub(_Z3RM, left.z3_expr, right.z3_expr)

class Mul(BinaryExpr):
    name: str = '*'

    def evaluate(self, ctx: EvalCtx):
        left, right = self.children
        return left.evaluate(ctx) * right.evaluate(ctx)

    def evaluate_sig(self, ctx: EvalCtx):
        left, right = self.children
        left_sig = left.evaluate_sig(ctx)
        right_sig = right.evaluate_sig(ctx)
        left_m, left_e = math.frexp(left_sig.x)
        right_m, right_e = math.frexp(right_sig.x)

        sigbits = min(left_e - left_sig.lsb, right_e - right_sig.lsb)
        maxbits = max(left_e - left_sig.lsb_min, right_e - right_sig.lsb_min)
        x = left_sig.x * right_sig.x
        m, e = math.frexp(x)

        return SigCtx(x, e - sigbits, e - maxbits)

    def _translate_z3(self):
        left, right = self.children
        return z3.fpMul(_Z3RM, left.z3_expr, right.z3_expr)

class Div(BinaryExpr):
    name: str = '/'

    def evaluate(self, ctx: EvalCtx):
        left, right = self.children
        return left.evaluate(ctx) / right.evaluate(ctx)

    def evaluate_sig(self, ctx: EvalCtx):
        left, right = self.children
        left_sig = left.evaluate_sig(ctx)
        right_sig = right.evaluate_sig(ctx)
        left_m, left_e = math.frexp(left_sig.x)
        right_m, right_e = math.frexp(right_sig.x)

        sigbits = min(left_e - left_sig.lsb, right_e - right_sig.lsb)
        maxbits = max(left_e - left_sig.lsb_min, right_e - right_sig.lsb_min)
        x = left_sig.x / right_sig.x
        m, e = math.frexp(x)

        out_sig = SigCtx(x, e - sigbits, e - maxbits)

        print("DIV")
        print(left_sig)
        print(right_sig)
        print(out_sig)
        print("")

        return out_sig

    def _translate_z3(self):
        left, right = self.children
        return z3.fpDiv(_Z3RM, left.z3_expr, right.z3_expr)


# comparison

_nary_lt = _impl_by_pairs(operator.lt, all)
_nary_gt = _impl_by_pairs(operator.gt, all)
_nary_le = _impl_by_pairs(operator.le, all)
_nary_ge = _impl_by_pairs(operator.ge, all)
_nary_eq = _impl_by_pairs(operator.eq, all)
_nary_ne = _impl_all_pairs(operator.ne, all)


class LT(NaryExpr):
    name: str = '<'

    def evaluate(self, ctx: EvalCtx):
        return _nary_lt(child.evaluate(ctx) for child in self.children)

class GT(NaryExpr):
    name: str = '>'

    def evaluate(self, ctx: EvalCtx):
        return _nary_gt(child.evaluate(ctx) for child in self.children)

class LEQ(NaryExpr):
    name: str = '<='

    def evaluate(self, ctx: EvalCtx):
        return _nary_le(child.evaluate(ctx) for child in self.children)

class GEQ(NaryExpr):
    name: str = '>='

    def evaluate(self, ctx: EvalCtx):
        return _nary_ge(child.evaluate(ctx) for child in self.children)

class EQ(NaryExpr):
    name: str = '=='

    def evaluate(self, ctx: EvalCtx):
        return _nary_eq(child.evaluate(ctx) for child in self.children)

class NEQ(NaryExpr):
    name: str = '!='

    def evaluate(self, ctx: EvalCtx):
        return _nary_ne(child.evaluate(ctx) for child in self.children)

# logic

class And(NaryExpr):
    name: str = 'and'

    def evaluate(self, ctx: EvalCtx):
        return all(*[child.evaluate(ctx) for child in self.children])

class Or(NaryExpr):
    name: str = 'or'

    def evaluate(self, ctx: EvalCtx):
        return any(*[child.evaluate(ctx) for child in self.children])

class Not(UnaryExpr):
    name: str = 'not'

    def evaluate(self, ctx: EvalCtx):
        (child,) = self.children
        return not child.evaluate(ctx)
