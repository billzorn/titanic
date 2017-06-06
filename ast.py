import operator
import math

import numpy as np
import gmpy2
from gmpy2 import mpfr
import z3

default_rm = z3.RoundNearestTiesToEven()

# main expression classes

class Expr(object):
    def __init__(self, *args):
        self.name = 'Expr'
        self.data = args
        self.op = None
        self.op_np = None
        self.op_mp = None
        self.op_z3 = None

    def _assert_nargs(self, n):
        if len(self.data) != n:
            raise ValueError('expected {} argument(s), got {}'.format(n, repr(self.data)))
        
    def __str__(self):
        return ('(' + self.name + ' {}'*len(self.data) + ')').format(*self.data)

    def __call__(self, argctx):
        return self.op(*(child(argctx) for child in self.data))

    def apply_np(self, argctx, sort):
        operation = self.op_np if self.op_np else self.op
        return operation(*(child.apply_np(argctx, sort) for child in self.data))

    def apply_mp(self, argctx, sort, toplevel = True):
        if toplevel:
            return _mp_toplevel(self, argctx, sort)
        else:
            operation = self.op_mp if self.op_mp else self.op
            return operation(*(child.apply_mp(argctx, sort, toplevel=False) for child in self.data))

    def apply_z3(self, argctx, sort, rm = default_rm):
        operation = self.op_z3 if self.op_z3 else self.op
        return operation(*(child.apply_z3(argctx, sort, rm=rm) for child in self.data))

class ExprRM(Expr):
    def apply_z3(self, argctx, sort, rm = default_rm):
        operation = self.op_z3 if self.op_z3 else self.op
        return operation(rm, *(child.apply_z3(argctx, sort, rm=rm) for child in self.data))

# base case values

# THERE WILL BE MORE STRING PARSING CODE HERE, AT LEAST FOR Z3 1.25*(2**42) STRINGS
np_sorts = {
    16 : np.float16,
    32 : np.float32,
    64 : np.float64,
    #128 : np.float128,
}
def np_val(data, sort):
    return np_sorts[sort](data)

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._assert_nargs(1)
        self.data = self.data[0]
        self.name = self.data
        self.op = float
        self.op_np = np_val
        self.op_mp = mpfr
        self.op_z3 = z3_val

    def __str__(self):
        return self.data

    def __call__(self, argctx):
        return self.op(self.data)

    def apply_np(self, argctx, sort):
        return self.op_np(self.data, sort)

    def apply_mp(self, argctx, sort, toplevel = True):
        return self.op_mp(self.data, sort)

    def apply_z3(self, argctx, sort, rm = default_rm):
        return self.op_z3(self.data, sort)

# strings as values are special: they cause a call to the value constructor
class Var(Val):
    def __call__(self, argctx):
        v = argctx[self.data]
        if isinstance(v, str):
            return self.op(v)
        else:
            return v

    def apply_np(self, argctx, sort):
        v = argctx[self.data]
        if isinstance(v, str):
            return self.op_np(v, sort)
        else:
            return v

    def apply_mp(self, argctx, sort, toplevel = True):
        v = argctx[self.data]
        if isinstance(v, str):
            return self.op_mp(v, sort)
        else:
            return v

    def apply_z3(self, argctx, sort, rm = default_rm):
        v = argctx[self.data]
        if isinstance(v, str):
            return self.op_z3(v, sort)
        else:
            return v

# arithmetic

class Add(ExprRM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._assert_nargs(2)
        self.name = '+'
        self.op = operator.add
        self.op_mp = gmpy2.add
        self.op_z3 = z3.fpAdd

class Sub(ExprRM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._assert_nargs(2)
        self.name = '-'
        self.op = operator.sub
        self.op_mp = gmpy2.sub
        self.op_z3 = z3.fpSub

class Mul(ExprRM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._assert_nargs(2)
        self.name = '*'
        self.op = operator.mul
        self.op_mp = gmpy2.mul
        self.op_z3 = z3.fpMul

class Div(ExprRM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._assert_nargs(2)
        self.name = '/'
        self.op = operator.truediv
        self.op_mp = gmpy2.div
        self.op_z3 = z3.fpDiv

class Sqrt(ExprRM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._assert_nargs(1)
        self.name = 'sqrt'
        self.op = math.sqrt
        self.op_np = np.sqrt
        self.op_mp = gmpy2.sqrt
        self.op_z3 = z3.fpSqrt

class Neg(Expr):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._assert_nargs(1)
        self.name = 'neg'
        self.op = operator.neg
        self.op_z3 = z3.fpNeg

# comparison
        
class LT(Expr):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._assert_nargs(2)
        self.name = '<'
        self.op = operator.lt
        self.op_z3 = z3.fpLT

class GT(Expr):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._assert_nargs(2)
        self.name = '>'
        self.op = operator.gt
        self.op_z3 = z3.fpGT

class LEQ(Expr):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._assert_nargs(2)
        self.name = '<='
        self.op = operator.le
        self.op_z3 = z3.fpLEQ

class GEQ(Expr):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._assert_nargs(2)
        self.name = '>='
        self.op = operator.ge
        self.op_z3 = z3.fpGEQ

class EQ(Expr):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._assert_nargs(2)
        self.name = '=='
        self.op = operator.eq
        self.op_z3 = z3.fpEQ

class NEQ(Expr):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._assert_nargs(2)
        self.name = '!='
        self.op = operator.ne
        self.op_z3 = z3.fpNEQ

# logic

def all_star(*args):
    return all(args)

def any_star(*args):
    return any(args)

class And(Expr):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'and'
        self.op = all_star
        self.op_z3 = z3.And

class Or(Expr):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'or'
        self.op = any_star
        self.op_z3 = z3.Or

class Not(Expr):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._assert_nargs(1)
        self.name = 'not'
        self.op = operator.not_
        self.op_z3 = z3.Not

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
