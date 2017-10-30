import operator
import math

import numpy as np
import z3

import real
FReal = real.FReal
import core

# local data type conversions

def fp_val(data):
    return float(data)

np_sorts = {
    16 : np.float16,
    32 : np.float32,
    64 : np.float64,
    # np.float128 is NOT compatible with IEEE 754 bianry128
}
def np_sort(sort):
    return np_sorts[sort]
def np_val(data, sort):
    return np_sorts[sort](data)

z3_sorts = {
    16 : z3.FPSort(5, 11),
    32 : z3.FPSort(8, 24),
    64 : z3.FPSort(11, 53),
}
def z3_sort(sort):
    if sort in z3_sorts:
        return z3_sorts[sort]
    else:
        return sort
def z3_val(data, sort):
    return z3.FPVal(data, z3_sort(sort))

real_sorts = {
    16 : (5, 11),
    32 : (8, 24),
    64 : (11, 53),
}
def real_sort(sort):
    if sort in real_sorts:
        return real_sorts[sort]
    else:
        return sort

def round_real(unrounded, sort, rm):
    if isinstance(unrounded, bool):
        return unrounded
    elif sort:
        w, p = real_sort(sort)
        return core.implicit_to_real(*core.real_to_implicit(unrounded, w, p, rm))
    else:
        return unrounded

# main expression classes

class Expr(object):
    name = 'Expr'
    nargs = None
    op = None
    op_np = None
    op_z3 = None
    op_real = None

    def __init__(self, *args):
        self.data = args
        if type(self).nargs is not None:
            self._assert_nargs(type(self).nargs)

    def _assert_nargs(self, n):
        if len(self.data) != n:
            raise ValueError('expected {} argument(s), got {}'.format(n, repr(self.data)))

    def __str__(self):
        return ('(' + type(self).name + ' {}'*len(self.data) + ')').format(*self.data)

    def __call__(self, argctx):
        return type(self).op(*(child(argctx) for child in self.data))

    def apply_np(self, argctx, sort):
        operation = type(self).op_np if type(self).op_np else type(self).op
        return operation(*(child.apply_np(argctx, sort) for child in self.data))

    def apply_z3(self, argctx, sort, rm):
        operation = type(self).op_z3 if type(self).op_z3 else type(self).op
        return operation(*(child.apply_z3(argctx, sort, rm) for child in self.data))

    def apply_real(self, argctx, sort, rm):
        operation = type(self).op_real if type(self).op_real else type(self).op
        unrounded = operation(*(child.apply_real(argctx, sort, rm) for child in self.data))
        return round_real(unrounded, sort, rm)

class ExprRM(Expr):
    name = 'ExprRM'

    def apply_z3(self, argctx, sort, rm):
        operation = type(self).op_z3 if type(self).op_z3 else type(self).op
        return operation(rm, *(child.apply_z3(argctx, sort, rm) for child in self.data))

# base case values

class Val(Expr):
    name = 'Val'
    nargs = 1
    op = fp_val
    op_np = np_val
    op_z3 = z3_val
    op_real = FReal

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = self.data[0]

    def __str__(self):
        return self.data

    def __call__(self, argctx):
        return type(self).op(self.data)

    def apply_np(self, argctx, sort):
        return type(self).op_np(self.data, sort)

    def apply_z3(self, argctx, sort, rm):
        return type(self).op_z3(self.data, sort)

    def apply_real(self, argctx, sort, rm):
        return round_real(type(self).op_real(self.data), sort, rm)

# strings as values are special: they cause a call to the value constructor
class Var(Val):
    name = 'Var'

    def __call__(self, argctx):
        v = argctx[self.data]
        if isinstance(v, str):
            return type(self).op(v)
        else:
            return v

    def apply_np(self, argctx, sort):
        v = argctx[self.data]
        if isinstance(v, str):
            return type(self).op_np(v, sort)
        else:
            return v

    def apply_z3(self, argctx, sort, rm):
        v = argctx[self.data]
        if isinstance(v, str):
            return type(self).op_z3(v, sort)
        else:
            return v

    def apply_real(self, argctx, sort, rm):
        v = argctx[self.data]
        if isinstance(v, str):
            return round_real(type(self).op_real(v), sort, rm)
        else:
            return v

# arithmetic

class Add(ExprRM):
    name = '+'
    nargs = 2
    op = operator.add
    op_z3 = z3.fpAdd

class Sub(ExprRM):
    name = '-'
    nargs = 2
    op = operator.sub
    op_z3 = z3.fpSub

class Mul(ExprRM):
    name = '*'
    nargs = 2
    op = operator.mul
    op_z3 = z3.fpMul

class Div(ExprRM):
    name = '/'
    nargs = 2
    op = operator.truediv
    op_z3 = z3.fpDiv

class Sqrt(ExprRM):
    name = 'sqrt'
    nargs = 1
    op = math.sqrt
    op_np = np.sqrt
    op_z3 = z3.fpSqrt
    op_real = real.sqrt

class Neg(Expr):
    name = 'neg'
    nargs = 1
    op = operator.neg
    op_z3 = z3.fpNeg

# comparison

class LT(Expr):
    name = '<'
    nargs = 2
    op = operator.lt
    op_z3 = z3.fpLT

class GT(Expr):
    name = '>'
    nargs = 2
    op = operator.gt
    op_z3 = z3.fpGT

class LEQ(Expr):
    name = '<='
    nargs = 2
    op = operator.le
    op_z3 = z3.fpLEQ

class GEQ(Expr):
    name = '>='
    nargs = 2
    op = operator.ge
    op_z3 = z3.fpGEQ

class EQ(Expr):
    name = '=='
    nargs = 2
    op = operator.eq
    op_z3 = z3.fpEQ

class NEQ(Expr):
    name = '!='
    nargs = 2
    op = operator.ne
    op_z3 = z3.fpNEQ

# logic

def all_star(*args):
    return all(args)

def any_star(*args):
    return any(args)

class And(Expr):
    name = 'and'
    op = all_star
    op_z3 = z3.And

class Or(Expr):
    name = 'or'
    op = any_star
    op_z3 = z3.Or

class Not(Expr):
    name = 'not'
    nargs = 1
    op = operator.not_
    op_z3 = z3.Not

if __name__ == '__main__':
    rm = 'RNE'
    z3rm = z3.RoundNearestTiesToEven()
    z3sort = z3.FPSort(8, 32)
    args = {'x' : '1', 'y' : 'pi', 'z' : z3.FP('z', z3sort),}
    print(args)

    a = Add(Val('0.32'), Var('x'))
    print(a)
    print(a(args))
    print(a.apply_np(args, 32))
    # print(a.apply_z3(args, 32, z3rm))
    print(a.apply_real(args, None, None))

    # something is messed up with Z3's printing...
    # b = Add(a, Var('z'))
    # print(b)
    # print(b.apply_z3(args, z3sort, z3rm))

    c = Add(a, Var('y'))
    print(c)
    print(c.apply_real(args, None, None))
    print(c.apply_real(args, 16, rm))
    print(c.apply_real(args, (5, 11), rm))
