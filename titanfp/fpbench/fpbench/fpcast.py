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


# Sinking Point library
import sinking as sp
Sink = sp.Sink
gmp = sp.gmp
mpfr = sp.mpfr


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

# evaluation contexts and modes

EVAL_754 = 0
EVAL_OPTIMISTIC = 1

class EvalCtx(object):
    
    def __init__(self, inputs, w=11, p=53, mode=EVAL_754):
        self.inputs = inputs
        self.variables = inputs
        self.mode = mode
        # 754-like
        self.w = w
        self.p = p
        self.emax = (1 << (self.w - 1)) - 1
        self.emin = 1 - self.emax
        self.n = self.emin - self.p

    def let(self, bindings):
        self.variables = self.variables.update(bindings)
        return self

# ast

class Expr(object):
    name: str = 'Expr'

    def __init__(self):
        self._z3_expr = None

    @property
    def z3_expr(self):
        if self._z3_expr is None:
            self._z3_expr = self._translate_z3()
        return self._z3_expr

    def __call__(self, *args, **kwargs):
        return self.evaluate(EvalCtx(*args, **kwargs))

    def evaluate(self, ctx: EvalCtx):
        raise ValueError('floating point unimplemented for ast node ' + type(self).__name__)

    def evaluate_sink(self, ctx: EvalCtx):
        raise ValueError('sinking point evaluation unimplemented for ast node ' + type(self).__name__)

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

    #TODO: any input processing beyond passing to float() / mpfr()
    
    def evaluate(self, ctx: EvalCtx):
        return float(self.value)

    def evaluate_sink(self, ctx: EvalCtx):
        rep = Sink(mpfr(self.value, ctx.p))
        if rep._n < ctx.n:
            prec = rep._e - min_n
            if prec < 2:
                raise ValueError('unsupported: gmp with precision < 2: {}({}), n={}, p={}'.format(
                    self.name, repr(self.value), repr(ctx.n), repr(ctx.p)))
            rep = Sink(mpfr(self.value, prec))
            if rep._n != ctx.n:
                #TODO: fixup
                # as in sp.withnprec
                if rep._n == min_n + 1:
                    # rounded up, put back one bit
                    rep = Sink(mpfr(result, result.precision + 1))
                else:
                    raise ValueError('unsupported: n should be {}, got {}: {}({}), n={}, p={}'.format(
                        repr(ctx.n), repr(rep._n), self.name, repr(self.value), repr(ctx.n), repr(ctx.p)))
        return rep

    def _translate_z3(self):
        return z3.FPVal(self.value, _Z3SORT)

class Var(ValueExpr):
    name: str = 'Var'

    def evaluate(self, ctx: EvalCtx):
        return ctx.variables[self.value]

    def evaluate_sink(self, ctx: EvalCtx):
        return ctx.variables[self.value]

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

    def evaluate_sink(self, ctx: EvalCtx):
        cond, then_body, else_body = self.children
        if cond.evaluate(ctx):
            return then_body.evaluate_sink(ctx)
        else:
            return else_body.evaluate_sink(ctx)

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

    def evaluate_sink(self, ctx: EvalCtx):
        (child,) = self.children
        # independent of mode
        return -child.evaluate_sink(ctx)

    def _translate_z3(self):
        (child,) = self.children
        return z3.fpNeg(_Z3RM, child.z3_expr)


class Sqrt(UnaryExpr):
    name: str = 'sqrt'

    def evaluate(self, ctx: EvalCtx):
        (child,) = self.children
        return math.sqrt(child.evaluate(ctx))

    def evaluate_sink(self, ctx: EvalCtx):
        (child,) = self.children
        rep = child.evaluate_sink(ctx)
        
        if ctx.mode == EVAL_754:
            n = ctx.n
            p = ctx.p
        elif ctx.mode == EVAL_OPTIMISTIC:
            n = ctx.n
            p = min(rep._p, ctx.p) if rep._inexact else ctx.p
        else:
            raise ValueError('unknown mode {}'.format(repr(ctx.mode)))

        result, op_inexact = sp.withnprec(gmp.sqrt, rep.as_mpfr(), min_n=n, max_p=p)

        if ctx.mode == EVAL_754:
            inexact = False
        elif ctx.mode == EVAL_OPTIMISTIC:
            inexact = rep._inexact or op_inexact
        else:
            raise ValueError('unknown mode {}'.format(repr(ctx.mode)))

        #TODO: envelopes?
        return Sink(result, inexact=inexact, sided_interval=False, full_interval=False)

    def _translate_z3(self):
        (child,) = self.children
        return z3.fpSqrt(_Z3RM, child.z3_expr)

class Add(BinaryExpr):
    name: str = '+'

    def evaluate(self, ctx: EvalCtx):
        left, right = self.children
        return left.evaluate(ctx) + right.evaluate(ctx)

    def evaluate_sink(self, ctx: EvalCtx):
        left, right = self.children
        left_rep = left.evaluate_sink(ctx)
        right_rep = right.evaluate_sink(ctx)

        if ctx.mode == EVAL_754:
            n = ctx.n
            p = ctx.p
        elif ctx.mode == EVAL_OPTIMISTIC:
            left_n = left_rep._n if left_rep._inexact else ctx.n
            right_n = right_rep._n if right_rep._inexact else ctx.n
            n = max(left_n, right_n)
            p = ctx.p
        else:
            raise ValueError('unknown mode {}'.format(repr(ctx.mode)))

        result, op_inexact = sp.withnprec(gmp.add, left_rep.as_mpfr(), right_rep.as_mpfr(), min_n=n, max_p=p)

        if ctx.mode == EVAL_754:
            inexact = False
        elif ctx.mode == EVAL_OPTIMISTIC:
            inexact = left_rep._inexact or right_rep._inexact or op_inexact
        else:
            raise ValueError('unknown mode {}'.format(repr(ctx.mode)))

        #TODO: envelopes?
        return Sink(result, inexact=inexact, sided_interval=False, full_interval=False)
        
    def _translate_z3(self):
        left, right = self.children
        return z3.fpAdd(_Z3RM, left.z3_expr, right.z3_expr)

class Sub(BinaryExpr):
    name: str = '-'

    def evaluate(self, ctx: EvalCtx):
        left, right = self.children
        return left.evaluate(ctx) - right.evaluate(ctx)

    def evaluate_sink(self, ctx: EvalCtx):
        left, right = self.children
        left_rep = left.evaluate_sink(ctx)
        right_rep = right.evaluate_sink(ctx)

        if ctx.mode == EVAL_754:
            n = ctx.n
            p = ctx.p
        elif ctx.mode == EVAL_OPTIMISTIC:
            left_n = left_rep._n if left_rep._inexact else ctx.n
            right_n = right_rep._n if right_rep._inexact else ctx.n
            n = max(left_n, right_n)
            p = ctx.p
        else:
            raise ValueError('unknown mode {}'.format(repr(ctx.mode)))

        result, op_inexact = sp.withnprec(gmp.sub, left_rep.as_mpfr(), right_rep.as_mpfr(), min_n=n, max_p=p)

        if ctx.mode == EVAL_754:
            inexact = False
        elif ctx.mode == EVAL_OPTIMISTIC:
            inexact = left_rep._inexact or right_rep._inexact or op_inexact
        else:
            raise ValueError('unknown mode {}'.format(repr(ctx.mode)))

        #TODO: envelopes?
        return Sink(result, inexact=inexact, sided_interval=False, full_interval=False)

    def _translate_z3(self):
        left, right = self.children
        return z3.fpSub(_Z3RM, left.z3_expr, right.z3_expr)

class Mul(BinaryExpr):
    name: str = '*'

    def evaluate(self, ctx: EvalCtx):
        left, right = self.children
        return left.evaluate(ctx) * right.evaluate(ctx)

    def evaluate_sink(self, ctx: EvalCtx):
        left, right = self.children
        left_rep = left.evaluate_sink(ctx)
        right_rep = right.evaluate_sink(ctx)

        if ctx.mode == EVAL_754:
            n = ctx.n
            p = ctx.p
        elif ctx.mode == EVAL_OPTIMISTIC:
            n = ctx.n
            left_p = left_rep._p if left_rep._inexact else ctx.p
            right_p = right_rep._p if right_rep._inexact else ctx.p
            p = min(left_p, right_p)
        else:
            raise ValueError('unknown mode {}'.format(repr(ctx.mode)))

        result, op_inexact = sp.withnprec(gmp.mul, left_rep.as_mpfr(), right_rep.as_mpfr(), min_n=n, max_p=p)

        if ctx.mode == EVAL_754:
            inexact = False
        elif ctx.mode == EVAL_OPTIMISTIC:
            inexact = left_rep._inexact or right_rep._inexact or op_inexact
        else:
            raise ValueError('unknown mode {}'.format(repr(ctx.mode)))

        #TODO: envelopes?
        return Sink(result, inexact=inexact, sided_interval=False, full_interval=False)

    def _translate_z3(self):
        left, right = self.children
        return z3.fpMul(_Z3RM, left.z3_expr, right.z3_expr)

class Div(BinaryExpr):
    name: str = '/'

    def evaluate(self, ctx: EvalCtx):
        left, right = self.children
        return left.evaluate(ctx) / right.evaluate(ctx)

    def evaluate_sink(self, ctx: EvalCtx):
        left, right = self.children
        left_rep = left.evaluate_sink(ctx)
        right_rep = right.evaluate_sink(ctx)

        if ctx.mode == EVAL_754:
            n = ctx.n
            p = ctx.p
        elif ctx.mode == EVAL_OPTIMISTIC:
            n = ctx.n
            left_p = left_rep._p if left_rep._inexact else ctx.p
            right_p = right_rep._p if right_rep._inexact else ctx.p
            p = min(left_p, right_p)
        else:
            raise ValueError('unknown mode {}'.format(repr(ctx.mode)))

        result, op_inexact = sp.withnprec(gmp.div, left_rep.as_mpfr(), right_rep.as_mpfr(), min_n=n, max_p=p)

        if ctx.mode == EVAL_754:
            inexact = False
        elif ctx.mode == EVAL_OPTIMISTIC:
            inexact = left_rep._inexact or right_rep._inexact or op_inexact
        else:
            raise ValueError('unknown mode {}'.format(repr(ctx.mode)))

        #TODO: envelopes?
        return Sink(result, inexact=inexact, sided_interval=False, full_interval=False)

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

    def evaluate_sink(self, ctx: EvalCtx):
        return _nary_lt(child.evaluate_sink(ctx) for child in self.children)

class GT(NaryExpr):
    name: str = '>'

    def evaluate(self, ctx: EvalCtx):
        return _nary_gt(child.evaluate(ctx) for child in self.children)

    def evaluate_sink(self, ctx: EvalCtx):
        return _nary_gt(child.evaluate_sink(ctx) for child in self.children)


class LEQ(NaryExpr):
    name: str = '<='

    def evaluate(self, ctx: EvalCtx):
        return _nary_le(child.evaluate(ctx) for child in self.children)

    def evaluate_sink(self, ctx: EvalCtx):
        return _nary_le(child.evaluate_sink(ctx) for child in self.children)

class GEQ(NaryExpr):
    name: str = '>='

    def evaluate(self, ctx: EvalCtx):
        return _nary_ge(child.evaluate(ctx) for child in self.children)

    def evaluate_sink(self, ctx: EvalCtx):
        return _nary_ge(child.evaluate_sink(ctx) for child in self.children)

class EQ(NaryExpr):
    name: str = '=='

    def evaluate(self, ctx: EvalCtx):
        return _nary_eq(child.evaluate(ctx) for child in self.children)

    def evaluate_sink(self, ctx: EvalCtx):
        return _nary_eq(child.evaluate_sink(ctx) for child in self.children)

class NEQ(NaryExpr):
    name: str = '!='

    def evaluate(self, ctx: EvalCtx):
        return _nary_ne(child.evaluate(ctx) for child in self.children)

    def evaluate_sink(self, ctx: EvalCtx):
        return _nary_ne(child.evaluate_sink(ctx) for child in self.children)

# logic

class And(NaryExpr):
    name: str = 'and'

    def evaluate(self, ctx: EvalCtx):
        return all(*[child.evaluate(ctx) for child in self.children])

    def evaluate_sink(self, ctx: EvalCtx):
        return all(*[child.evaluate_sink(ctx) for child in self.children])

class Or(NaryExpr):
    name: str = 'or'

    def evaluate(self, ctx: EvalCtx):
        return any(*[child.evaluate(ctx) for child in self.children])

    def evaluate_sink(self, ctx: EvalCtx):
        return any(*[child.evaluate_sink(ctx) for child in self.children])

class Not(UnaryExpr):
    name: str = 'not'

    def evaluate(self, ctx: EvalCtx):
        (child,) = self.children
        return not child.evaluate(ctx)

    def evaluate_sink(self, ctx: EvalCtx):
        (child,) = self.children
        return not child.evaluate_sink(ctx)
