import numpy as np
import gmpy2
from gmpy2 import mpfr
import z3

default_rm = z3.RoundNearestTiesToEven()

# variable names

arglo_str = '_arglo_'
arghi_str = '_arghi_'
argsep = '_'
def arglo(i, name):
    return '{}{:d}{}{}'.format(arglo_str, i, argsep, name)
def arghi(i, name):
    return '{}{:d}{}{}'.format(arghi_str, i, argsep, name)
def get_arglo(s):
    if s.startswith(arglo_str):
        i_name = s[len(arglo_str):].split(argsep)
        return int(i_name[0]), argsep.join(i_name[1:])
    else:
        return None, None
def get_arghi(s):
    if s.startswith(arghi_str):
        i_name = s[len(arghi_str):].split(argsep)
        return int(i_name[0]), argsep.join(i_name[1:])
    else:
        return None, None
reslo = 'lo_result'
reshi = 'hi_result'
resexp = 'expected_result'

# conversion of strings to values

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
def z3_sort(sort):
    if sort in z3_sorts:
        return z3_sorts[sort]
    else:
        return sort
def z3_val(data, sort):
    return z3.FPVal(data, z3_sort(sort))

# temporary! - restricts prescision to at most float64
z3fp_constants = {
    '+oo' : float('+inf'),
    '-oo' : float('-inf'),
    'NaN' : float('nan'),
}
def get_z3fp(v):
    if v in z3fp_constants:
        return z3fp_constants[v]
    else:
        return float(eval(v))

# symbolic units in the last place difference

def z3fp_to_ordinal(x, sort, rm = default_rm):
    z3_sort = z3_sorts[sort] if sort in z3_sorts else sort
    x_prime = z3.fpToFP(rm, x, z3sort) if x.sort() != z3sort else x 
    return z3.If(x_prime < z3.FPVal(0.0, x_prime.sort()),
                 -z3.fpToIEEEBV(-x_prime),
                 z3.fpToIEEEBV(z3.fpAbs(x_prime)))

def z3ulps(x, y, sort, rm = default_rm):
    xz = z3fp_to_ordinal(x, sort, rm=rm)
    yz = z3fp_to_ordinal(y, sort, rm=rm)
    return z3.If(xz < yz, yz - xz, xz - yz)

