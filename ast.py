import operator
import math

import numpy as np
import gmpy2
from gmpy2 import mpfr
import z3

default_rm = z3.RoundNearestTiesToEven()

# main expression classes

class Expr(object):
    name = 'Expr'
    nargs = None
    op = None
    op_np = None
    op_mp = None
    op_z3 = None
    
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

    def apply_mp(self, argctx, sort, toplevel = True):
        if toplevel:
            return _mp_toplevel(self, argctx, sort)
        else:
            operation = type(self).op_mp if type(self).op_mp else type(self).op
            return operation(*(child.apply_mp(argctx, sort, toplevel=False) for child in self.data))

    def apply_z3(self, argctx, sort, rm = default_rm):
        operation = type(self).op_z3 if type(self).op_z3 else type(self).op
        return operation(*(child.apply_z3(argctx, sort, rm=rm) for child in self.data))

class ExprRM(Expr):
    name = 'ExprRM'

    def apply_z3(self, argctx, sort, rm = default_rm):
        operation = type(self).op_z3 if type(self).op_z3 else type(self).op
        return operation(rm, *(child.apply_z3(argctx, sort, rm=rm) for child in self.data))

# base case values

# THERE WILL BE MORE STRING PARSING CODE HERE, AT LEAST FOR Z3 1.25*(2**42) STRINGS
def fp_val(data):
    return float(data)

np_sorts = {
    16 : np.float16,
    32 : np.float32,
    64 : np.float64,
    #128 : np.float128,
}
def np_val(data, sort):
    return np_sorts[sort](data)

def mp_val(data, sort):
    return mpfr(data, precision=sort)

z3_sorts = {
    16 : z3.FPSort(5, 11),
    32 : z3.FPSort(8, 24),
    64 : z3.FPSort(11, 53),
    #128 : 
}
def z3_val(data, sort):
    if sort in z3_sorts:
        return z3.FPVal(data, z3_sorts[sort])
    else:
        return z3.FPVal(data, sort)

class Val(Expr):
    name = 'Val'
    nargs = 1
    op = fp_val
    op_np = np_val
    op_mp = mp_val
    op_z3 = z3_val
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = self.data[0]

    def __str__(self):
        return self.data

    def __call__(self, argctx):
        return type(self).op(self.data)

    def apply_np(self, argctx, sort):
        return type(self).op_np(self.data, sort)

    def apply_mp(self, argctx, sort, toplevel = True):
        return type(self).op_mp(self.data, sort)

    def apply_z3(self, argctx, sort, rm = default_rm):
        return type(self).op_z3(self.data, sort)

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

    def apply_mp(self, argctx, sort, toplevel = True):
        v = argctx[self.data]
        if isinstance(v, str):
            return type(self).op_mp(v, sort)
        else:
            return v

    def apply_z3(self, argctx, sort, rm = default_rm):
        v = argctx[self.data]
        if isinstance(v, str):
            return type(self).op_z3(v, sort)
        else:
            return v

# arithmetic

class Add(ExprRM):
    name = '+'
    nargs = 2
    op = operator.add
    op_mp = gmpy2.add
    op_z3 = z3.fpAdd

class Sub(ExprRM):
    name = '-'
    nargs = 2
    op = operator.sub
    op_mp = gmpy2.sub
    op_z3 = z3.fpSub

class Mul(ExprRM):
    name = '*'
    nargs = 2
    op = operator.mul
    op_mp = gmpy2.mul
    op_z3 = z3.fpMul

class Div(ExprRM):
    name = '/'
    nargs = 2
    op = operator.truediv
    op_mp = gmpy2.div
    op_z3 = z3.fpDiv

class Sqrt(ExprRM):
    name = 'sqrt'
    nargs = 1
    op = math.sqrt
    op_np = np.sqrt
    op_mp = gmpy2.sqrt
    op_z3 = z3.fpSqrt

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

# converts the given sort into context information, reruns computation until convergence is achieved for that sort
def _mp_toplevel(e, argctx, sort):
    # convergence is not implemented yet
    with gmpy2.local_context(gmpy2.context(), precision=sort) as ctx:
        return e.apply_mp(argctx, sort, toplevel=False)


if __name__ == '__main__':
    z3sort = z3.FPSort(8, 32)
    args = {'x' : '1', 'z' : z3.FP('z', z3sort)}
    print(args)
    
    a = Add(Val('0.32'), Var('x'))
    print(a)
    print(a(args))
    print(a.apply_np(args, 32))
    print(a.apply_mp(args, 32))
    print(a.apply_z3(args, 32))

    b = Add(a, Var('z'))
    print(b)
    print(b.apply_z3(args, z3sort))
