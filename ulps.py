import time
import math
import z3
import gmpy2
from gmpy2 import mpfr

z3.set_option(max_args=10000000, max_lines=1000000, max_depth=10000000, max_visited=1000000)
print('using z3 {}'.format(z3.get_version_string()))

import sexp

# some definitiony things
FP16 = z3.FPSort(5, 11)
FP32 = z3.FPSort(8, 24)
FP64 = z3.FPSort(11, 53)
RTZ = z3.RoundTowardZero()
RNE = z3.RoundNearestTiesToEven()

default_fplo = FP16
default_fphi = FP32
default_maxulps = (2 ** (default_fplo.ebits() + default_fplo.sbits())) - 1
default_mpfr_ctx = 4096

# Convenient sort conversion.
def fp_up(fpnum):
    return z3.fpToFP(RNE, fpnum, default_fphi)
def fp_down(fpnum):
    return z3.fpToFP(RNE, fpnum, default_fplo)

z3fp_constants = {
    '+oo' : math.inf,
    '-oo' : -math.inf,
    'NaN' : math.nan,
}
def get_z3fp(v):
    if v in z3fp_constants:
        return z3fp_constants[v]
    else:
        return float(eval(v))

# Total ordering of FP numbers.
def fp_to_ordinal(x):
    return z3.If(x < z3.FPVal(0.0, x.sort()),
                 -z3.fpToIEEEBV(-x),
                 z3.fpToIEEEBV(z3.fpAbs(x)))

# Absolute value of units last place difference.
def ulps(x, y):
    xz = fp_to_ordinal(x)
    yz = fp_to_ordinal(y)
    return z3.If(xz < yz, yz - xz, xz - yz)

# Common names.
def arglo(i):
    return 'arg_{:d}_lo_'.format(i)
def arghi(i):
    return 'arg_{:d}_hi_'.format(i)
result_lo = 'r32'
result_hi = 'r64'
result_cast = 'c32'
result_ulps = 'u'

# -- OUTDATED: use new sexp_solve() to pass expressions as fpbench cores --
# Set up arguments, upcasts, results, and downcast. Don't run the solver yet.
def dual_solve(expr, nargs, min_input = None, min_open = True, max_input = None, max_open = True,
               fplo = default_fplo, fphi = default_fphi, lotohi = fp_up, hitolo = fp_down):
    args32 = []
    args64 = []
    for i in range(nargs):
        args32.append(z3.FP(arglo(i), fplo))
        args64.append(z3.FP(arghi(i), fphi))

    r32 = z3.FP(result_lo, fplo)
    r64 = z3.FP(result_hi, fphi)
    c32 = z3.FP(result_cast, fplo)
    s = z3.Solver()
    
    for arg32, arg64 in zip(args32, args64):
        if min_input is not None:
            if min_open:
                s.add(z3.FPVal(min_input, arg32.sort()) <= arg32)
            else:
                s.add(z3.FPVal(min_input, arg32.sort()) < arg32)
        if max_input is not None:
            if max_open:
                s.add(arg32 <= z3.FPVal(max_input, arg32.sort()))
            else:
                s.add(arg32 < z3.FPVal(max_input, arg32.sort()))
        s.add(arg64 == lotohi(arg32))
        
    s.add(r32 == expr(*args32))
    s.add(r64 == expr(*args64))
    s.add(c32 == fp_down(r64))

    return s, r32, c32

# expr is a (sexp string) fpcore.
def sexp_solve(expr, nargs, min_input = None, min_open = True, max_input = None, max_open = True,
               fplo = default_fplo, fphi = default_fphi, lotohi = fp_up, hitolo = fp_down):
    
    # no longer needed
    # args32 = []
    # args64 = []
    # for i in range(nargs):
    #     args32.append(z3.FP(arglo(i), fplo))
    #     args64.append(z3.FP(arghi(i), fphi))

    fpcore_arguments, fpcore_e, fpcore_properties = sexp.fpcore_loads(expr)
    # assert nargs == len(fpcore_arguments)

    args32 = {}
    args64 = {}
    for i, argname in enumerate(fpcore_arguments):
        z3arg32 = z3.FP(arglo(i) + argname, fplo)
        args32[argname] = z3arg32
        z3arg64 = z3.FP(arghi(i) + argname, fphi)
        args64[argname] = z3arg64

    r32 = z3.FP(result_lo, fplo)
    r64 = z3.FP(result_hi, fphi)
    c32 = z3.FP(result_cast, fplo)
    s = z3.Solver()
    
    for argname in args32:
        z3arg32 = args32[argname]
        z3arg64 = args64[argname]
        # no longer needed
        # if min_input is not None:
        #     if min_open:
        #         s.add(z3.FPVal(min_input, z3arg32.sort()) <= z3arg32)
        #     else:
        #         s.add(z3.FPVal(min_input, z3arg32.sort()) < z3arg32)
        # if max_input is not None:
        #     if max_open:
        #         s.add(z3arg32 <= z3.FPVal(max_input, z3arg32.sort()))
        #     else:
        #         s.add(z3arg32 < z3.FPVal(max_input, z3arg32.sort()))
        s.add(z3arg64 == lotohi(z3arg32))

    #print(fpcore_properties)
    if ':pre' in fpcore_properties:
        #print('pre is {}'.format(fpcore_properties[':pre']))
        s.add(sexp.construct_expr(fpcore_properties[':pre'], args32, 'z3', rm=RNE))

    s.add(r32 == sexp.construct_expr(fpcore_e, args32, 'z3', rm=RNE))
    s.add(r64 == sexp.construct_expr(fpcore_e, args64, 'z3', rm=RNE))
    s.add(c32 == fp_down(r64))

    return s, r32, c32


# One result is NaN, and the other isn't.
def with_NaN(expr, nargs, min_input = None, min_open = True, max_input = None, max_open = True,
             fplo = default_fplo, fphi = default_fphi, lotohi = fp_up, hitolo = fp_down):

    s, r32, c32 = sexp_solve(expr, nargs,
                             min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open,
                             fplo=fplo, fphi=fphi, lotohi=lotohi, hitolo=hitolo)

    s.add(z3.Or(z3.And(z3.fpIsNaN(r32), z3.Not(z3.fpIsNaN(c32))),
                z3.And(z3.Not(z3.fpIsNaN(r32)), z3.fpIsNaN(c32))))
    return s

# The low precision result is NaN.
def with_NaN_r(expr, nargs, min_input = None, min_open = True, max_input = None, max_open = True,
               fplo = default_fplo, fphi = default_fphi, lotohi = fp_up, hitolo = fp_down):

    s, r32, c32 = sexp_solve(expr, nargs,
                             min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open,
                             fplo=fplo, fphi=fphi, lotohi=lotohi, hitolo=hitolo)

    s.add(z3.And(z3.fpIsNaN(r32), z3.Not(z3.fpIsNaN(c32))))
    return s

# The high precision result is NaN.
def with_NaN_c(expr, nargs, min_input = None, min_open = True, max_input = None, max_open = True,
               fplo = default_fplo, fphi = default_fphi, lotohi = fp_up, hitolo = fp_down):

    s, r32, c32 = sexp_solve(expr, nargs,
                             min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open,
                             fplo=fplo, fphi=fphi, lotohi=lotohi, hitolo=hitolo)

    s.add(z3.And(z3.Not(z3.fpIsNaN(r32)), z3.fpIsNaN(c32)))
    return s

# One result is an infinity, and the results are not equal.
def with_Inf(expr, nargs, min_input = None, min_open = True, max_input = None, max_open = True,
             fplo = default_fplo, fphi = default_fphi, lotohi = fp_up, hitolo = fp_down):

    s, r32, c32 = sexp_solve(expr, nargs,
                             min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open,
                             fplo=fplo, fphi=fphi, lotohi=lotohi, hitolo=hitolo)

    s.add(z3.And(z3.Or(z3.fpIsInf(r32), z3.fpIsInf(c32)),
                 z3.Not(c32 == r32)))
    return s

# The low precision result is positive infinity.
def with_Inf_r_pos(expr, nargs, min_input = None, min_open = True, max_input = None, max_open = True,
                   fplo = default_fplo, fphi = default_fphi, lotohi = fp_up, hitolo = fp_down):

    s, r32, c32 = sexp_solve(expr, nargs, min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open,
                             fplo=fplo, fphi=fphi, lotohi=lotohi, hitolo=hitolo)

    s.add(z3.And(z3.fpIsInf(r32), z3.fpIsPositive(r32), z3.Not(c32 == r32)))
    return s

# The low precision result is negative infinity.
def with_Inf_r_neg(expr, nargs, min_input = None, min_open = True, max_input = None, max_open = True,
                   fplo = default_fplo, fphi = default_fphi, lotohi = fp_up, hitolo = fp_down):

    s, r32, c32 = sexp_solve(expr, nargs,
                             min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open,
                             fplo=fplo, fphi=fphi, lotohi=lotohi, hitolo=hitolo)

    s.add(z3.And(z3.fpIsInf(r32), z3.fpIsNegative(r32), z3.Not(c32 == r32)))
    return s

# The high precision result is positive infinity.
def with_Inf_c_pos(expr, nargs, min_input = None, min_open = True, max_input = None, max_open = True,
                   fplo = default_fplo, fphi = default_fphi, lotohi = fp_up, hitolo = fp_down):

    s, r32, c32 = sexp_solve(expr, nargs,
                             min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open,
                             fplo=fplo, fphi=fphi, lotohi=lotohi, hitolo=hitolo)

    s.add(z3.And(z3.fpIsInf(c32), z3.fpIsPositive(c32), z3.Not(c32 == r32)))
    return s

# The high precision result is negative infinity.
def with_Inf_c_neg(expr, nargs, min_input = None, min_open = True, max_input = None, max_open = True,
                   fplo = default_fplo, fphi = default_fphi, lotohi = fp_up, hitolo = fp_down):

    s, r32, c32 = sexp_solve(expr, nargs,
                             min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open,
                             fplo=fplo, fphi=fphi, lotohi=lotohi, hitolo=hitolo)

    s.add(z3.And(z3.fpIsInf(c32), z3.fpIsNegative(c32), z3.Not(c32 == r32)))
    return s

# Both results are zero, but the signs are not equal.
def with_zeros(expr, nargs, min_input = None, min_open = True, max_input = None, max_open = True,
               fplo = default_fplo, fphi = default_fphi, lotohi = fp_up, hitolo = fp_down):

    s, r32, c32 = sexp_solve(expr, nargs,
                             min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open,
                             fplo=fplo, fphi=fphi, lotohi=lotohi, hitolo=hitolo)

    s.add(z3.Or(z3.And(r32 == z3.FPVal(0.0, r32.sort()), c32 == z3.FPVal(-0.0, c32.sort())),
                z3.And(r32 == z3.FPVal(-0.0, r32.sort()), c32 == z3.FPVal(0.0, c32.sort()))))
    return s

# The low precision result is positive zero.
def with_zeros_r_pos(expr, nargs, min_input = None, min_open = True, max_input = None, max_open = True,
                     fplo = default_fplo, fphi = default_fphi, lotohi = fp_up, hitolo = fp_down):

    s, r32, c32 = sexp_solve(expr, nargs,
                             min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open,
                             fplo=fplo, fphi=fphi, lotohi=lotohi, hitolo=hitolo)

    s.add(z3.And(r32 == z3.FPVal(0.0, r32.sort()), c32 == z3.FPVal(-0.0, c32.sort())))
    return s

# The low precision result is negative zero.
def with_zeros_r_neg(expr, nargs, min_input = None, min_open = True, max_input = None, max_open = True,
                     fplo = default_fplo, fphi = default_fphi, lotohi = fp_up, hitolo = fp_down):

    s, r32, c32 = sexp_solve(expr, nargs,
                             min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open,
                             fplo=fplo, fphi=fphi, lotohi=lotohi, hitolo=hitolo)

    s.add(z3.And(r32 == z3.FPVal(-0.0, r32.sort()), c32 == z3.FPVal(0.0, c32.sort())))
    return s

# Main query: find inputs with at least the requested number of ulps.
def with_ulps(expr, nargs, min_ulps, min_input = None, min_open = True, max_input = None, max_open = True,
              fplo = default_fplo, fphi = default_fphi, lotohi = fp_up, hitolo = fp_down):

    s, r32, c32 = sexp_solve(expr, nargs,
                             min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open,
                             fplo=fplo, fphi=fphi, lotohi=lotohi, hitolo=hitolo)
    u = z3.BitVec('u', r32.ebits() + r32.sbits())

    # we have independent checks for all bad behavior involving NaN, Inf, and mismatched signs at zero.
    # ulps inherently treat +-zero the same, so we only need explicit conditions about NaN and Inf
    s.add(z3.Not(z3.fpIsNaN(r32)))
    s.add(z3.Not(z3.fpIsNaN(c32)))
    s.add(z3.Not(z3.fpIsInf(r32)))
    s.add(z3.Not(z3.fpIsInf(c32)))

    s.add(u == ulps(r32, c32))
    s.add(z3.UGE(u, z3.BitVecVal(min_ulps, u.size())))

    return s

# Simple execution and reporting.

def serialize_model(s):
    try:
        m = s.model()
        return {str(e) : str(m[e]) for e in m}
    except z3.z3types.Z3Exception:
        return None
        # this has the slightly annoying disadvantage of misreporting 'unknown'

def serialize_stats(s):
    stats = s.statistics()
    return {k : v for k, v in stats}

def model_get_by_name(m, name):
    # try:
    #     if name in m:
    #         return m[name]
    #     else:
    #         return None
    # except Exception:
    #     for e in m:
    #         if str(e) == name:
    #             return m[e]
    #     return None
    for e in m:
        if str(e).startswith(name):
            return str(e), m[e]
    return None, None

def model_get_fp_by_name(m, name):
    for e in m:
        if str(e).startswith(name):
            return str(e), get_z3fp(m[e])
    return None, None
    

def run_query(s, descr):
    if descr:
        print('Running check for {}.'.format(descr))
    return s.check()

def report_query(query, descr, m, nargs, stats, elapsed):
    print('Report for {}: ({:.2f}s)'.format(descr, elapsed))
    if m:
        args = {}
        print('  arguments:')
        for i in range(nargs):
            argname = arglo(i)
            name, v = model_get_fp_by_name(m, arglo(i))
            shortname = name[len(argname):]
            args[shortname] = v
            print('    {} = {}'.format(shortname, v))            
        print('  low precision: {}'.format(model_get_fp_by_name(m, result_lo)[1]))
        print('  exp precision: {}'.format(model_get_fp_by_name(m, result_cast)[1]))
        print('  full precision: {}'.format(model_get_fp_by_name(m, result_hi)[1]))

        # mpfr goooo!
        pcore_arguments, fpcore_e, fpcore_properties = sexp.fpcore_loads(query)

        with gmpy2.local_context(gmpy2.context(), precision=default_mpfr_ctx) as ctx:
            mpfrgs = {k : mpfr(args[k]) for k in args}
            mpfresult = sexp.construct_expr(fpcore_e, mpfrgs, 'mpfr')
            print('  mpfr {}: {}'.format(default_mpfr_ctx, repr(mpfresult)))

        
        ulps = model_get_by_name(m, result_ulps)[1]
        if ulps is not None:
            print('  {} ulps'.format(ulps))
    else:
        print('  unsat')
    if stats:
        print('  statistics:')
        for k in ['conflicts', 'decisions', 'max memory', 'restarts']:
            if k in stats:
                print('    {}: {}'.format(k, stats[k]))
    
def run_and_report(query, s, args, descr, print_statistics = True):
    #print(s.sexpr())
    start = time.time()
    status = run_query(s, descr)
    end = time.time()
    m = serialize_model(s)
    if print_statistics:
        stats = serialize_stats(s)
    else:
        stats = None
    report_query(query, descr, m, args, stats, end - start)
    if m:
        ulps = model_get_by_name(m, result_ulps)[1]
        if ulps is not None:
            return int(ulps)
        else:
            return None
    else:
        return None


def binsearch_ulps(expr, nargs, min_input = None, min_open = True, max_input = None, max_open = True):
    ulps_target = 1
    ulps_lo = 1
    ulps_hi = None
    ulps_scale = 2

    print('\nBinsearch: looking for ulps_hi')
    
    # find ulps_hi
    while ulps_hi is None:
        print('\nlo={}, target={}, hi={}'.format(ulps_lo, ulps_target, ulps_hi))
        ulps = run_and_report(expr, with_ulps(expr, nargs, ulps_target,
                                              min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open),
                              nargs, '>= {:d} ulps'.format(ulps_target))
        if ulps is None:
            ulps_hi = ulps_target # we got unsat: can't get this many ulps
        else:
            ulps_lo = ulps # we have a cxx of exactly this many ulps: could be worse than target
            ulps_target = min(ulps_lo * ulps_scale, default_maxulps)

    print('\nBinsearch: narrowing')
            
    # binsearch
    while ulps_hi > ulps_lo + 1:
        ulps_target = ((ulps_hi - ulps_lo) // 2) + ulps_lo
        print('\nlo={}, target={}, hi={}'.format(ulps_lo, ulps_target, ulps_hi))
        ulps = run_and_report(expr, with_ulps(expr, nargs, ulps_target,
                                        min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open),
                              nargs, '>= {:d} ulps'.format(ulps_target))
        if ulps is None:
            ulps_hi = ulps_target
        else:
            ulps_lo = ulps

    print('\nBinsearch finished.')
    print('  The largest assignment we could find had {} ulps.'.format(ulps_lo))
    print('  Could not find an assignment for {} ulps.'.format(ulps_hi))


def run_all(expr, nargs, ulps, min_input = None, min_open = True, max_input = None, max_open = True):
    run_and_report(expr, with_NaN(expr, nargs,
                                   min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open,),
                   nargs, 'NaN')
    return
    run_and_report(expr, with_Inf(expr, nargs,
                                   min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open,),
                   nargs, 'Inf')
    run_and_report(expr, with_zeros(expr, nargs,
                                     min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open,),
                   nargs, 'zeros')
    run_and_report(expr, with_ulps(expr, nargs, ulps,
                                    min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open),
                   nargs, '>= {:d} ulps'.format(ulps))

def run_more(expr, nargs, min_input = None, min_open = True, max_input = None, max_open = True):
    run_and_report(expr, with_NaN_r(expr, nargs,
                                     min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open,),
                   nargs, 'r=NaN')
    run_and_report(expr, with_NaN_c(expr, nargs,
                                     min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open,),
                   nargs, 'c=NaN')
    run_and_report(expr, with_Inf_r_pos(expr, nargs,
                                         min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open,),
                   nargs, 'r=+Inf')
    run_and_report(expr, with_Inf_r_neg(expr, nargs,
                                         min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open,),
                   nargs, 'r=-Inf')
    run_and_report(expr, with_Inf_c_pos(expr, nargs,
                                         min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open,),
                   nargs, 'c=+Inf')
    run_and_report(expr, with_Inf_c_neg(expr, nargs,
                                         min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open,),
                   nargs, 'c=-Inf')
    run_and_report(expr, with_zeros_r_pos(expr, nargs,
                                           min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open,),
                   nargs, 'r=+0.0, c=-0.0')
    run_and_report(expr, with_zeros_r_neg(expr, nargs,
                                    min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open,),
                   nargs, 'r=-0.0, c=+0.0')
    binsearch_ulps(expr, nargs, min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open)
    
# DEMO

# sqrt(-1 + x*x)
def expr1(x, rm = RNE):
    return z3.fpSqrt(rm, z3.fpAdd(rm, z3.FPVal(-1.0, x.sort()), z3.fpMul(rm, x, x)))

expr1_fpcore = '''(FPCore 
 (x)
 :name "sqrt(x^2 - 1)"
 :pre (>= x 1)
 (sqrt (- (* x x) 1)))
'''

fpc2 = '''(FPCore
 (a b c)
 :name "NMSE p42, positive"
 :cite (hamming-1987)
 :pre (and (>= (* b b) (* 4 (* a c))) (!= a 0))
 (/ (+ (- b) (sqrt (- (* b b) (* 4 (* a c))))) (* 2 a)))
'''

fpc_rosa = '''(FPCore
 (x)
 :name "Rosa's Benchmark"
 :cite (darulova-kuncak-2014)
 (- (* 0.954929658551372 x) (* 0.12900613773279798 (* (* x x) x))))
'''

fpc_hamming_3_6 = '''(FPCore
 (x)
 :name "NMSE example 3.6"
 :cite (hamming-1987)
 :pre (>= x 0)
 (- (/ 1 (sqrt x)) (/ 1 (sqrt (+ x 1)))))
'''

fpc_turbine = '''(FPCore
 (v w r)
 :name "Rosa's TurbineBenchmark"
 :cite (darulova-kuncak-2014)
 (- (- (+ 3 (/ 2 (* r r))) (/ (* (* 0.125 (- 3 (* 2 v))) (* (* (* w w) r) r)) (- 1 v))) 4.5))'''

fpc_hamming_3_1 = '''(FPCore
 (x)
 :name "NMSE example 3.1"
 :cite (hamming-1987)
 :pre (>= x 0)
 (- (sqrt (+ x 1)) (sqrt x)))
'''


if __name__ == '__main__':
    # go
    fpc = fpc_hamming_3_1
    print(fpc)
    run_more(fpc, 1)


# # test ulps
# xv = 30.0
# yv = 30.0001
# x = z3.BitVec('x', 32)
# y = z3.BitVec('y', 32)
# xz = z3.BitVec('xz', 32)
# yz = z3.BitVec('yz', 32)
# u = z3.BitVec('u', 32)
# s.add(x == z3.fpToIEEEBV(z3.FPVal(xv, FP32)))
# s.add(y == z3.fpToIEEEBV(z3.FPVal(yv, FP32)))
# s.add(xz == fp_to_ordinal(z3.FPVal(xv, FP32)))
# s.add(yz == fp_to_ordinal(z3.FPVal(yv, FP32)))
# s.add(u == ulps(z3.FPVal(xv, FP32), z3.FPVal(yv, FP32)))
# print(s.check())
# print(s.model())
