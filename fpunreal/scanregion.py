import numpy as np

import mathlib
from fpcore import FPCore

def scanregion(core, inputs, n, sort):
    all_points = []
    for varname in inputs:
        v = inputs[varname]
        # < v
        pre_points = []
        new_v = v
        new_inputs = {k : inputs[k] for k in inputs}
        for i in range(n):
            new_v = mathlib.npfp_prev(new_v, sort)
            new_inputs[varname] = str(new_v)
            lo_result = core.expr.apply_np(new_inputs, sort)
            hi_result = core.expr.apply_mp(new_inputs, 8192)
            exp_result = mathlib.np_val(hi_result, sort)

            ulps = mathlib.npulps(lo_result, exp_result, sort)

            pre_points.append((varname, new_v, lo_result, exp_result, ulps))

        # = v
        new_inputs[varname] = str(v)
        lo_result = core.expr.apply_np(new_inputs, sort)
        hi_result = core.expr.apply_mp(new_inputs, 8192)
        exp_result = mathlib.np_val(hi_result, sort)

        ulps = mathlib.npulps(lo_result, exp_result, sort)

        center_point = [('!' + varname, v, lo_result, exp_result, ulps)]

        # > v
        post_points = []
        new_v = v
        for i in range(n):
            new_v = mathlib.npfp_next(new_v, sort)
            new_inputs[varname] = str(new_v)
            lo_result = core.expr.apply_np(new_inputs, sort)
            hi_result = core.expr.apply_mp(new_inputs, 8192)
            exp_result = mathlib.np_val(hi_result, sort)

            ulps = mathlib.npulps(lo_result, exp_result, sort)

            post_points.append((varname, new_v, lo_result, exp_result, ulps))

        # add to list
        pre_points.reverse()
        all_points.append(pre_points + center_point + post_points)
    return all_points

def printregion(all_points):
    for scan in all_points:
        for varname, v, lo_result, exp_result, ulps in scan:
            print('{:3s}: {:24s} -> {:16s} {:16s} {:10d} ulps'.format(varname, str(v), str(lo_result), str(exp_result), ulps))
        print()


fpc = '''(FPCore
 (x)
 :name "NMSE example 3.1"
 :cite (hamming-1987)
 :pre (>= x 0)
 (- (sqrt (+ x 1)) (sqrt x)))
'''

fpc2 = '''(FPCore
 (a b c)
 :name "NMSE p42, positive"
 :cite (hamming-1987)
 :pre (and (>= (* b b) (* 4 (* a c))) (!= a 0))
 (/ (+ (- b) (sqrt (- (* b b) (* 4 (* a c))))) (* 2 a)))
'''

core = FPCore(fpc)

pts = scanregion(core, {'x': '1024.0'}, 10, 16)
print(fpc)
printregion(pts)

core2 = FPCore(fpc2)

pts2 = scanregion(core2, {'a': '-5.960464477539063e-08',
                          'b': '0.00017261505126953125',
                          'c': '0.5',}, 10, 16)
print(fpc2)
printregion(pts2)



args = {'a': '-1.5046329484187314e-36',
        'b': '-0.0',
        'c': '3.231394258733028e-42',}
print(args)
printregion(scanregion(core2, args, 15, 32))


args = {'a': '-5.605193857299268e-45',
        'b': '7.343140135242376e-23',
        'c': '-0.12498092651367188',}
print(args)
printregion(scanregion(core2, args, 15, 32))



fpc_turbine = '''(FPCore
 (v w r)
 :name "Rosa's TurbineBenchmark"
 :cite (darulova-kuncak-2014)
 (- (- (+ 3 (/ 2 (* r r))) (/ (* (* 0.125 (- 3 (* 2 v))) (* (* (* w w) r) r)) (- 1 v))) 4.5))
'''

