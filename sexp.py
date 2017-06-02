from numbers import Number
import math
import operator

import sexpdata
import z3
import gmpy2
from gmpy2 import mpfr

RNE = z3.RoundNearestTiesToEven()

Sym = sexpdata.Symbol
fpcore_constants = {
    'E'        : math.e,
    'LOG2E'    : None,
    'LOG10E'   : None,
    'LN2'      : None,
    'LN10'     : None,
    'PI'       : math.pi,
    'PI_2'     : None,
    'PI_4'     : None,
    '1_PI'     : math.pi,
    '2_PI'     : math.tau,
    '2_SQRTPI' : None,
    'SQRT2'    : None,
    'SQRT1_2'  : None,
    'INFINITY' : math.inf,
    'NAN'      : math.nan,
}
fpcore_modes = {
    'z3'   : 0,
    'mpfr' : 1,
    'native': 2,
}

def opand(x, y):
    return x and y
def opor(x, y):
    return x or y

def nconj(bconj):
    def conj(*args):
        if len(args) > 2:
            return bconj(args[0], conj(*args[1:]))
        else:
            return bconj(*args)
    return conj

def impl_by_pairs(op, conj):
    def impl(*args):
        if len(args) > 2:
            return conj(*(op(args[i], args[i+1]) for i in range(len(args)-1)))
        else:
            return op(*args)
    return impl

def impl_all_pairs(op, conj):
    def impl(*args):
        if len(args) > 2:
            pairs = []
            for i in range(len(args)-1):
                pairs += [op(args[i], args[j]) for j in range(i+1,len(args))]
            return conj(*pairs)
        else:
            return op(*args)    
    return impl

fpcore_operations = {
    # math
    '+'    : (z3.fpAdd, gmpy2.add, operator.add,),
    '-'    : (z3.fpSub, gmpy2.sub, operator.sub,),
    '*'    : (z3.fpMul, gmpy2.mul, operator.mul,),
    '/'    : (z3.fpDiv, gmpy2.div, operator.truediv,),
    'sqrt' : (z3.fpSqrt, gmpy2.sqrt, math.sqrt,),
    # SPECIAL UNARY MATH
    'UNARY-' : (z3.fpNeg, lambda x: gmpy2.sub(mpfr(0), x),),
    # comparison
    '<'  : (impl_by_pairs(z3.fpLT, z3.And), impl_by_pairs(operator.lt, nconj(opand)), impl_by_pairs(operator.lt, nconj(opand)),),
    '>'  : (impl_by_pairs(z3.fpGT, z3.And), impl_by_pairs(operator.gt, nconj(opand)), impl_by_pairs(operator.gt, nconj(opand)),),
    '<=' : (impl_by_pairs(z3.fpLEQ, z3.And), impl_by_pairs(operator.le, nconj(opand)), impl_by_pairs(operator.le, nconj(opand)),),
    '>=' : (impl_by_pairs(z3.fpGEQ, z3.And), impl_by_pairs(operator.ge, nconj(opand)), impl_by_pairs(operator.ge, nconj(opand)),),
    '==' : (impl_by_pairs(z3.fpEQ, z3.And), impl_by_pairs(operator.eq, nconj(opand)), impl_by_pairs(operator.eq, nconj(opand)),),
    '!=' : (impl_all_pairs(z3.fpNEQ, z3.And), impl_all_pairs(operator.ne, nconj(opand)), impl_by_pairs(operator.ne, nconj(opand)),),
    # logic
    'and' : (z3.And, nconj(opand), nconj(opand),),
    'or'  : (z3.Or, nconj(opor), nconj(opor),),
    'not' : (z3.Not, operator.not_, operator.not_,),
}
needs_z3_rm = {
    '+',
    '-',
    '*',
    '/',
    'sqrt',
}

def is_parens(e):
    return isinstance(e, list)
def is_number(e):
    return isinstance(e, Number)
def is_constant(e):
    return isinstance(e, Sym) and str(e) in fpcore_constants
def is_symbol(e):
    return isinstance(e, Sym) and str(e) not in fpcore_constants

def get_number(e):
    return float(e)
def get_constant(e):
    x = fpcore_constants[str(e)]
    if x is None:
        raise ValueError('constant {} is unimplemented'.format(str(e)))
    else:
        return x
def get_symbol(e):
    return e.value()

def mode_value(c, mode, args):
    if mode == 'z3':
        return z3.FPVal(c, next(iter(args.values())).sort())
    elif mode == 'mpfr':
        return mpfr(c)
    else:
        return float(c)

def construct_expr(e, args, mode, rm = RNE):
    # ( operation expr* )
    # ( if expr expr expr )
    # ( let ( [ symbol expr ]* ) expr )
    # ( while expr ( [ symbol expr expr ]* ) expr 
    if is_parens(e):
        op = get_symbol(e[0])
        es = e[1:]
        # check for unary -
        if op == '-' and len(es) == 1:
            op = 'UNARY-'
        # unimplemented
        if op in {'if', 'let', 'while'}:
            raise ValueError('op {} is unimplemented'.format(op))
        elif op not in fpcore_operations:
            raise ValueError('unknown operation {}'.format(op))
        else:
            f = fpcore_operations[op][fpcore_modes[mode]]
            fargs = [construct_expr(x, args, mode, rm=rm) for x in es]
            if mode == 'z3' and op in needs_z3_rm:
                fargs = [rm] + fargs
            return f(*fargs)
    # number
    elif is_number(e):
        c = get_number(e)
        return mode_value(c, mode, args)
    # constant
    elif is_constant(e):
        c = get_constant(e)
        return mode_value(c, mode, args)
    # symbol
    elif is_symbol(e):
        return args[get_symbol(e)]

def extract_components(core):
    fpcore_type = core[0]
    fpcore_arguments = core[1]
    fpcore_e = core[-1]
    properties = core[2:-1]
    fpcore_properties = {get_symbol(properties[i]) : properties[i+1]
                         for i in range(0, len(properties), 2)}
    assert fpcore_type == sexpdata.Symbol('FPCore')
    return fpcore_arguments, fpcore_e, fpcore_properties

def fpcore_loads(s):
    core = sexpdata.loads(s)
    fpcore_arguments, fpcore_e, fpcore_properties = extract_components(core)
    return [get_symbol(a) for a in fpcore_arguments], fpcore_e, fpcore_properties

if __name__ == '__main__':

    def ople(x, y):
        return '({} < {})'.format(x, y)
    def opand(x, y):
        return '({} and {})'.format(x, y)

    impl = impl_by_pairs(ople, nconj(opand))
    impl_all = impl_all_pairs(ople, nconj(opand))

    print(impl('x', 'y', 'z', 'w'))
    print(impl_all('x', 'y', 'z', 'w'))
    print(impl('a', 'b'))
    print(impl_all('a', 'b'))
