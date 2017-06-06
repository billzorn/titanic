import numpy as np
import gmpy2
from gmpy2 import mpfr
import z3

default_rm = z3.RoundNearestTiesToEven()

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
def z3_val(data, sort):
    if sort in z3_sorts:
        return z3.FPVal(data, z3_sorts[sort])
    else:
        return z3.FPVal(data, sort)

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

