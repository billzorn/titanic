import struct

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
    z3sort = z3_sort(sort)
    x_prime = z3.fpToFP(rm, x, z3sort) if x.sort() != z3sort else x 
    return z3.If(x_prime < z3.FPVal(0.0, x_prime.sort()),
                 -z3.fpToIEEEBV(-x_prime),
                 z3.fpToIEEEBV(z3.fpAbs(x_prime)))

def z3ulps(x, y, sort, rm = default_rm):
    xz = z3fp_to_ordinal(x, sort, rm=rm)
    yz = z3fp_to_ordinal(y, sort, rm=rm)
    return z3.If(xz < yz, yz - xz, xz - yz)


def np_bytes_to_int(a):
    n = 0
    for i, x in enumerate(a):
        n |= x << (i * 8)
    return n

def np_int_to_bytes(x, n):
    return bytes((x >> i) & 0xff for i in range(0, n, 8))

def npfp_to_int(x):
    return np_bytes_to_int(x.tobytes())

def npfp_from_int(x):
    pass

def bitstr(x, sort):
    npsort = np_sorts[sort]
    bits = ('{:0' + str(sort) + 'b}').format(np_bytes_to_int(npsort(x).tobytes()))
    exp_mant = 1 + np.finfo(npsort).nexp
    return ','.join((bits[:1], bits[1:exp_mant], bits[exp_mant:]))

# 0.0 and -0.0 both map to 0
def npfp_to_ordinal(x, sort):
    npsort = np_sorts[sort]
    x_prime = npsort(x)
    if x_prime < 0.0:
        return -np_bytes_to_int((-x_prime).tobytes())
    else:
        return np_bytes_to_int(np.abs(x_prime).tobytes())

# cannot produce -0.0
def npfp_from_ordinal(x, sort):
    npsort = np_sorts[sort]
    if x < 0:
        return None

def npulps(x, y, sort):
    xz = npfp_to_ordinal(x, sort)
    yz = npfp_to_ordinal(y, sort)
    if xz < yz:
        return yz - xz
    else:
        return xz - yz

def npfp_next(x, sort):
    npsort = np_sorts[sort]
    return np.nextafter(npsort(x), npsort('+inf'))

def npfp_next0(x, sort):
    npsort = np_sorts[sort]
    if npsort(x) == npsort('0.0') and np.copysign(npsort('1.0'), npsort(x)) == npsort('-1.0'):
        return npsort('0.0')
    return np.nextafter(npsort(x), npsort('+inf'))

def npfp_prev(x, sort):
    npsort = np_sorts[sort]
    return np.nextafter(npsort(x), npsort('-inf'))

def npfp_prev0(x, sort):
    npsort = np_sorts[sort]
    if npsort(x) == npsort('0.0') and np.copysign(npsort('1.0'), npsort(x)) == npsort('1.0'):
        return npsort('-0.0')
    return np.nextafter(npsort(x), npsort('-inf'))


def npfp_tiny(sort):
    npsort = np_sorts[sort]
    return np.finfo(npsort).tiny

def npfp_inf(sort):
    npsort = np_sorts[sort]
    return npsort('inf')

def npfp_one(sort):
    npsort = np_sorts[sort]
    return npsort('1.0')

def npfp_zero(sort):
    npsort = np_sorts[sort]
    return npsort('0.0')

if __name__ == '__main__':    
    print(npulps(0, 1, 16))
    print(npulps(-1, 0, 32))
