import math
import operator

from .fpbench import fpcparser
from .arithmetic import mpmf, ieee754, evalctx, analysis
from .arithmetic.mpmf import Interpreter

from .sweep import search


sqrt_core = '''(FPCore sqrt_bfloat_limit (a residual_bound)
 :precision {overall_prec}

 (while* (and (! :titanic-analysis skip (< steps (# 10))) (>= (fabs residual) residual_bound))
  ([x a (! :precision {diff_prec} (- x
                                     (! :precision {scale_prec} (/ residual (* 2 x)))))]
   [residual (! :precision {res_prec} (- (* x x) a))
             (! :precision {res_prec} (- (* x x) a))]
   [steps (! :titanic-analysis skip (# 0))
          (! :titanic-analysis skip (# (+ 1 steps)))])
  (cast x))

)
'''

sqrt_bab_core = '''(FPCore bab_bfloat_limit (a residual_bound)
 :precision {overall_prec}

 (while* (and (! :titanic-analysis skip (< steps (# 10))) (>= (fabs residual) residual_bound))
  ([x a (! :precision {diff_prec} (* 1/2 (+ x
                                            (! :precision {scale_prec} (/ a x)))))]
   [residual (! :precision {res_prec} (- (* x x) a))
             (! :precision {res_prec} (- (* x x) a))]
   [steps (! :titanic-analysis skip (# 0))
          (! :titanic-analysis skip (# (+ 1 steps)))])
  (cast x)
 )

)
'''


def linear_ulps(x, y):
    smaller_n = min(x.n, y.n)
    x_offset = x.n - smaller_n
    y_offset = y.n - smaller_n

    x_c = x.c << x_offset
    y_c = y.c << y_offset

    return x_c - y_c


formatted = sqrt_core.format(
    overall_prec = '(float 8 16)',
    res_prec = '(float 5 16)',
    diff_prec = '(float 5 16)',
    scale_prec = '(float 5 16)',
)

evaltor = Interpreter()
core = fpcparser.compile1(formatted)

bf16 = ieee754.ieee_ctx(8,16)
f64 = ieee754.ieee_ctx(11,64)
one = mpmf.MPMF(1, ctx=bf16)
ten = mpmf.MPMF(10, ctx=bf16)
current = one
one_to_ten = []
reference_results = []
while current <= ten:
    one_to_ten.append(current)
    reference = float(str(current))
    reference_results.append((current.sqrt(ctx=bf16), math.sqrt(reference)))
    current = current.next_float()

print(f'{len(one_to_ten)!s} bfloat16 test cases between 1 and 10 inclusive')

# def run_tests(core, bound):

#     results = []

#     for arg, ref in zip(one_to_ten, reference_results):
#         evaltor = Interpreter()
#         als, bc_als = analysis.DefaultAnalysis(), analysis.BitcostAnalysis()
#         #evaltor.analyses = [als, bc_als]
#         result = evaltor.interpret(core, (arg, bound))
#         err = (ref.sub(result)).fabs()
#         steps = evaltor.evals

#         results.append((str(arg), str(result), str(err), steps))

#     return results

# def oob(results):
#     counted = 0
#     maxerr = 0
#     sumerr = 0
#     sumcount = 0
#     for arg, result, err, steps in results:
#         if steps > 200:
#             counted += 1
#         abserr = abs(float(err))
#         if math.isfinite(abserr):
#             sumerr += abserr
#             sumcount += 1
#         # if abserr > 0.01:
#         #     print(arg, result, err, steps)
#         if abserr > maxerr:
#             maxerr = abserr
#     # print(f'{counted!s} inputs ran for more than 200 steps.')
#     # print(f'worst point was {maxerr!s}.')
#     return counted, maxerr, sumerr / sumcount


# new, with bitcost
def run_and_eval_bitcost(core, bound):

    timeouts = 0
    infs = 0
    worst_abserr = 0
    total_abserr = 0
    worst_ulps = 0
    total_ulps = 0
    worst_bitcost = 0
    total_bitcost = 0

    for arg, (ref_lo, ref_hi) in zip(one_to_ten, reference_results):
        evaltor = Interpreter()
        als = analysis.BitcostAnalysis()
        evaltor.analyses = [als]
        result = evaltor.interpret(core, (arg, bound))

        err = ref_hi - float(str(result))
        ulps = linear_ulps(result, ref_lo)
        steps = evaltor.evals
        bitcost = als.bits_requested

        abserr = abs(err)
        absulps = abs(ulps)

        if steps > 200:
            timeouts += 1

        if math.isfinite(abserr):
            if abserr > worst_abserr:
                worst_abserr = abserr
            total_abserr += abserr
            if absulps > worst_ulps:
                worst_ulps = absulps
            total_ulps += absulps
        else:
            worst_abserr = math.inf
            total_abserr = math.inf
            worst_ulps = math.inf
            total_ulps = math.inf
            infs += 1

        if bitcost > worst_bitcost:
            worst_bitcost = bitcost
        total_bitcost += bitcost

    return timeouts, infs, worst_abserr, total_abserr, worst_ulps, total_ulps, worst_bitcost, total_bitcost



def sweep_stage(bound, expbits, res_bits, diff_bits, scale_bits):

    res_nbits = expbits + res_bits
    diff_nbits = expbits + diff_bits
    scale_nbits = expbits + scale_bits

    formatted = sqrt_core.format(
        overall_prec = '(float 8 16)',
        res_prec = f'(float {expbits!s} {res_nbits!s})',
        diff_prec = f'(float {expbits!s} {diff_nbits!s})',
        scale_prec = f'(float {expbits!s} {scale_nbits!s})',
    )
    core = fpcparser.compile1(formatted)

    return run_and_eval_bitcost(core, bound)

    # results = run_tests(core, bound)
    # #counted, worst, avg = oob(results)
    # return oob(results)



def bab_stage(bound, expbits, res_bits, diff_bits, scale_bits):

    res_nbits = expbits + res_bits
    diff_nbits = expbits + diff_bits
    scale_nbits = expbits + scale_bits

    formatted = sqrt_bab_core.format(
        overall_prec = '(float 8 16)',
        res_prec = f'(float {expbits!s} {res_nbits!s})',
        diff_prec = f'(float {expbits!s} {diff_nbits!s})',
        scale_prec = f'(float {expbits!s} {scale_nbits!s})',
    )
    core = fpcparser.compile1(formatted)

    return run_and_eval_bitcost(core, bound)


# New, to facilitate using search interface
import random

def init_bound():
    return 1/100

def neighbor_bound(x):
    yield 1/100

def init_expbits():
    return random.randint(3,8)

def neighbor_expbits(x):
    nearby = 1
    for neighbor in range(x-nearby, x+nearby+1):
        if 1 <= neighbor <= 8 and neighbor != x:
            yield neighbor

def init_p():
    return random.randint(1, 16)

def neighbor_p(x):
    nearby = 2
    for neighbor in range(x-nearby, x+nearby+1):
        if 1 <= neighbor <= 32 and neighbor != x:
            yield neighbor


newton_inits = [
    init_bound,
    init_expbits,
    init_p,
    init_p,
    init_p
]

newton_neighbors = [
    neighbor_bound,
    neighbor_expbits,
    neighbor_p,
    neighbor_p,
    neighbor_p
]

newton_metrics = (operator.lt,) * 8

def run_random():
    return search.sweep_random_init(bab_stage, newton_inits, newton_neighbors, newton_metrics)


from multiprocessing import Pool


def sweep(stage_fn):
    bound = 1/100

    minexp = 3
    maxexp = 5
    minp = 2
    maxp = 20

    # minexp = 3
    # maxexp = 4
    # minp = 12
    # maxp = 14

    cfgs = 0

    frontier = []
    metrics = (operator.lt,) * 8

    with Pool() as p:
        result_buf = []

        print('building async result list')

        for expbits in range(minexp,maxexp+1):
            for res_bits in range(minp, maxp+1):
                for diff_bits in range(minp, maxp+1):
                    for scale_bits in range(minp, maxp+1):
                        args = bound, expbits, res_bits, diff_bits, scale_bits
                        result_buf.append((args, p.apply_async(stage_fn, args)))

        print(f'waiting for {len(result_buf)!s} results to come back')

        for args, result in result_buf:
            config = args
            bound, expbits, res_bits, diff_bits, scale_bits = config
            result_get = result.get()
            timeouts, infs, worst_abserr, total_abserr, worst_ulps, total_ulps, worst_bitcost, total_bitcost = result_get

            cfgs += 1
            print(f' -- {cfgs!s} -- ran {config!r}, got {result_get!r}')

            if True: # timeouts == 0 and infs == 0:
                frontier_elt = (config, result_get)

                updated, frontier = search.update_frontier(frontier, frontier_elt, metrics)
                if updated:
                    print('New frontier:')
                    search.print_frontier(frontier)

    print(f'tried {cfgs!s} configs, done\n')
    print('Final frontier:')
    search.print_frontier(frontier)
    print(flush=True)

    return frontier


def go():

    frontier_newton = sweep(sweep_stage)
    frontier_babylonian = sweep(bab_stage)

    print('\n\n\n\n')
    print('final frontier for newton:')
    search.print_frontier(frontier_newton)
    print('final frontier for babylonian:')
    search.print_frontier(frontier_babylonian)

    return frontier_newton, frontier_babylonian


"""
(FPCore sum ((A n))
  (for ([i n])
    ([accum 0 (+ accum (ref A i))])
    accum))

(FPCore ksum ((A n))
 :name "Kahan Summation"
  (for* ([i n])
    ([y 0 (- (ref A i) c)]
     [t 0 (+ accum y)]
     [c 0 (- (- t accum) y)]
     [accum 0 t])
    accum))

(FPCore nksum ((A n))
 :name "Neumaier's improved Kahan Summation algorithm"
  (for* ([i n])
    ([elt 0 (ref A i)]
     [t 0 (+ accum elt)]
     [c 0 (if (>= (fabs accum) (fabs elt))
              (+ c (+ (- accum t) elt))
              (+ c (+ (- elt t) accum)))]
     [accum 0 t])
    (+ accum c)))


(FPCore addpairs ((A n))
 :pre (> n 1)
  (tensor ([i (# (/ (+ n 1) 2))])
    (let* ([k1 (# (* i 2))]
           [k2 (# (+ k1 1))])
      (if (< k2 n)
          (+ (ref A k1) (ref A k2))
          (ref A k1)))
  ))

(FPCore binsum ((A n))
  (while (> (size B 0) 1)
    ([B A (addpairs B)])
    (if (== (size B 0) 0) 0 (ref B 0))))

(FPCore binsum-inline ((A n))
  (while (> (size B 0) 1)
    ([B A (tensor ([i (# (/ (+ (size B 0) 1) 2))])
            (let* ([k1 (# (* i 2))]
                   [k2 (# (+ k1 1))])
              (if (< k2 (size B 0))
                  (+ (ref B k1) (ref B k2))
                  (ref B k1)))
          )])
    (if (== (size B 0) 0) 0 (ref B 0))))


(FPCore dotprod ((A n) (B m))
 :pre (== n m)
  (for ([i n])
    ([accum 0 (+ accum (* (ref A i) (ref B i)))])
    accum))

(FPCore dotprod-fused ((A n) (B m))
 :pre (== n m)
  (for ([i n])
    ([accum 0 (fma (ref A i) (ref B i) accum)])
    accum))


(FPCore vec-prod ((A n) (B m))
 :pre (== n m)
  (tensor ([i n])
    (* (ref A i) (ref B i))))

(FPCore dotprod-kahan ((A n) (B m))
 :pre (== n m)
  (ksum (vec-prod A B)))

(FPCore dotprod-neumaier ((A n) (B m))
 :pre (== n m)
  (nksum (vec-prod A B)))

(FPCore dotprod-bin ((A n) (B m))
 :pre (== n m)
  (binsum (vec-prod A B)))

(FPCore main ((A n) (B m))
  (dotprod-bin A B))
"""




# TODO:
# sqrt:
# - sweep script
# - filtering

# dotprod:
# - implement quire sizing?
# - make test set
# - sweep:
#   - linear
#   - pairwise
#   - compensated ?

# RK:
# - get baselines
# - add other equations
# - average???

# img:
# - get base image
# - run




# there are 5 metrics:

# how many examples timed out
# how many examples reported inf
# the worst error (not residual!) of an example
# the worst individual bitcost
# the total bitcost (sum of examples)

# And here are the frontiers:

# tried 11375 configs, done

# Final frontier:
# {
#   [4, 14, 15, 8] : (0, 0, 0.010319312515875811, 2722, 583906)
#   [4, 15, 15, 8] : (0, 0, 0.009583499933366824, 2309, 586235)
#   [4, 15, 15, 9] : (0, 0, 0.009225947422650371, 2324, 589008)
#   [4, 15, 15, 10] : (0, 0, 0.009225947422650371, 1900, 592208)
#   [4, 15, 16, 9] : (0, 0, 0.008814576415149933, 2342, 592590)
#   [4, 15, 16, 10] : (0, 0, 0.008814576415149933, 1914, 596226)
#   [4, 15, 16, 11] : (0, 0, 0.008540409355627165, 1926, 599862)
#   [4, 16, 15, 10] : (0, 0, 0.008913438213184577, 1914, 595794)
# }
# Final bounded frontier:
# {
#   [4, 15, 15, 8] : (0, 0, 0.009583499933366824, 2309, 586235)
#   [4, 15, 15, 9] : (0, 0, 0.009225947422650371, 2324, 589008)
#   [4, 15, 15, 10] : (0, 0, 0.009225947422650371, 1900, 592208)
#   [4, 15, 16, 9] : (0, 0, 0.008814576415149933, 2342, 592590)
#   [4, 15, 16, 10] : (0, 0, 0.008814576415149933, 1914, 596226)
#   [4, 15, 16, 11] : (0, 0, 0.008540409355627165, 1926, 599862)
#   [4, 16, 15, 10] : (0, 0, 0.008913438213184577, 1914, 595794)
# }

# ([([4, 14, 15, 8], (0, 0, 0.010319312515875811, 2722, 583906)), ([4, 15, 15, 8], (0, 0, 0.009583499933366824, 2309, 586235)), ([4, 15, 15, 9], (0, 0, 0.009225947422650371, 2324, 589008)), ([4, 15, 15, 10], (0, 0, 0.009225947422650371, 1900, 592208)), ([4, 15, 16, 9], (0, 0, 0.008814576415149933, 2342, 592590)), ([4, 15, 16, 10], (0, 0, 0.008814576415149933, 1914, 596226)), ([4, 15, 16, 11], (0, 0, 0.008540409355627165, 1926, 599862)), ([4, 16, 15, 10], (0, 0, 0.008913438213184577, 1914, 595794))], [([4, 15, 15, 8], (0, 0, 0.009583499933366824, 2309, 586235)), ([4, 15, 15, 9], (0, 0, 0.009225947422650371, 2324, 589008)), ([4, 15, 15, 10], (0, 0, 0.009225947422650371, 1900, 592208)), ([4, 15, 16, 9], (0, 0, 0.008814576415149933, 2342, 592590)), ([4, 15, 16, 10], (0, 0, 0.008814576415149933, 1914, 596226)), ([4, 15, 16, 11], (0, 0, 0.008540409355627165, 1926, 599862)), ([4, 16, 15, 10], (0, 0, 0.008913438213184577, 1914, 595794))])




# # larger search space, for newton and babylonian

# final frontier for newton:
# [
#     ([4, 10, 11, 4], (0, 0, 0.010319312515875811, 2722, 583906)),
#     ([4, 11, 11, 4], (0, 0, 0.009583499933366824, 2309, 586235)),
#     ([4, 11, 11, 5], (0, 0, 0.009225947422650371, 2324, 589008)),
#     ([4, 11, 11, 6], (0, 0, 0.009225947422650371, 1900, 592208)),
#     ([4, 11, 12, 5], (0, 0, 0.008814576415149933, 2342, 592590)),
#     ([4, 11, 12, 6], (0, 0, 0.008814576415149933, 1914, 596226)),
#     ([4, 11, 12, 7], (0, 0, 0.008540409355627165, 1926, 599862)),
#     ([4, 12, 11, 6], (0, 0, 0.008913438213184577, 1914, 595794)),
# ]
# final frontier for babylonian:
# [
#     ([3, 8, 10, 11], (0, 0, 0.00504269614631192, 1876, 579698)),
#     ([3, 9, 11, 11], (0, 0, 0.004352769716506222, 2350, 596824)),
#     ([3, 9, 11, 12], (0, 0, 0.004352769716506222, 1912, 598042)),
#     ([3, 10, 11, 10], (0, 0, 0.004691951575984676, 2357, 596218)),
# ]
# ([([4, 10, 11, 4], (0, 0, 0.010319312515875811, 2722, 583906)), ([4, 11, 11, 4], (0, 0, 0.009583499933366824, 2309, 586235)), ([4, 11, 11, 5], (0, 0, 0.009225947422650371, 2324, 589008)), ([4, 11, 11, 6], (0, 0, 0.009225947422650371, 1900, 592208)), ([4, 11, 12, 5], (0, 0, 0.008814576415149933, 2342, 592590)), ([4, 11, 12, 6], (0, 0, 0.008814576415149933, 1914, 596226)), ([4, 11, 12, 7], (0, 0, 0.008540409355627165, 1926, 599862)), ([4, 12, 11, 6], (0, 0, 0.008913438213184577, 1914, 595794))], [([3, 8, 10, 11], (0, 0, 0.00504269614631192, 1876, 579698)), ([3, 9, 11, 11], (0, 0, 0.004352769716506222, 2350, 596824)), ([3, 9, 11, 12], (0,
# 0, 0.004352769716506222, 1912, 598042)), ([3, 10, 11, 10], (0, 0, 0.004691951575984676, 2357, 596218))])





# # big sweep with more metrics:

# final frontier for newton:
# [
#     ((0.01, 3, 9, 9, 7), (0, 161, inf, inf, inf, inf, 598, 191192)),
#     ((0.01, 4, 8, 11, 6), (3, 0, 0.010381265149109531, 1.242087232483563, 1, 52, 1802, 260348)),
#     ((0.01, 4, 8, 12, 6), (2, 0, 0.010381265149109531, 1.2298544386134243, 1, 54, 1840, 261730)),
#     ((0.01, 4, 9, 8, 5), (147, 0, 0.009773524781915288, 1.1503324101758734, 1, 31, 1689, 387749)),
#     ((0.01, 4, 9, 8, 6), (143, 0, 0.009773524781915288, 1.138653417748136, 1, 22, 1719, 390557)),
#     ((0.01, 4, 9, 8, 7), (143, 0, 0.008444389121245788, 1.1367809910672564, 1, 20, 1749, 397583)),
#     ((0.01, 4, 9, 8, 8), (143, 0, 0.008269696968013385, 1.1349352114328546, 1, 18, 1779, 403436)),
#     ((0.01, 4, 9, 12, 5), (1, 0, 0.009178976435896935, 1.1948248298679525, 1, 40, 1841, 262741)),
#     ((0.01, 4, 9, 13, 7), (1, 0, 0.00889602653105559, 1.1915233171778479, 1, 48, 1232, 273296)),
#     ((0.01, 4, 10, 8, 5), (151, 0, 0.009178976435896935, 1.1476254091218527, 1, 28, 1720, 398140)),
#     ((0.01, 4, 10, 8, 6), (149, 0, 0.008550528229663179, 1.1351608106749524, 1, 20, 1750, 402534)),
#     ((0.01, 4, 10, 8, 7), (149, 0, 0.008550528229663179, 1.1355080375801843, 1, 17, 1780, 409665)),
#     ((0.01, 4, 10, 8, 8), (149, 0, 0.008550528229663179, 1.1316429297171735, 1, 14, 1810, 415950)),
#     ((0.01, 4, 10, 11, 4), (1, 0, 0.010125389025929188, 1.2376857880900332, 1, 60, 1154, 258146)),
#     ((0.01, 4, 10, 11, 6), (1, 0, 0.009018450247308074, 1.1916867733028231, 1, 48, 1864, 265240)),
#     ((0.01, 4, 10, 11, 7), (0, 0, 0.009773524781915288, 1.1931248480991623, 1, 43, 1025, 268394)),
#     ((0.01, 4, 10, 12, 6), (2, 0, 0.009018450247308074, 1.1767367693664237, 1, 38, 1902, 270160)),
#     ((0.01, 4, 10, 12, 9), (3, 0, 0.009018450247308074, 1.1754944083988537, 1, 33, 1992, 281709)),
#     ((0.01, 4, 10, 13, 6), (1, 0, 0.008550528229663179, 1.173195460074455, 1, 35, 1940, 272956)),
#     ((0.01, 4, 10, 13, 10), (0, 0, 0.009010855143516405, 1.1727837992109647, 1, 32, 1106, 285806)),
#     ((0.01, 4, 10, 13, 11), (1, 0, 0.009372866688597714, 1.1716936808853893, 1, 31, 2090, 289790)),
#     ((0.01, 4, 10, 13, 12), (1, 0, 0.009010855143516405, 1.1684705373108841, 1, 29, 2120, 293626)),
#     ((0.01, 4, 10, 13, 13), (0, 0, 0.009010855143516405, 1.1681393169590697, 1, 30, 1151, 295633)),
#     ((0.01, 4, 10, 13, 14), (0, 0, 0.009010855143516405, 1.1687916720176317, 1, 29, 1166, 299454)),
#     ((0.01, 4, 10, 14, 9), (0, 0, 0.009010855143516405, 1.1753850044459737, 1, 31, 1109, 286184)),
#     ((0.01, 4, 10, 14, 10), (0, 0, 0.009010855143516405, 1.1701421488694357, 1, 33, 1124, 289808)),
#     ((0.01, 4, 10, 14, 11), (0, 0, 0.009372866688597714, 1.172097089796289, 1, 31, 1139, 292604)),
#     ((0.01, 4, 11, 8, 6), (143, 0, 0.008550528229663179, 1.1313471810792373, 1, 15, 1781, 403621)),
#     ((0.01, 4, 11, 8, 7), (143, 0, 0.00794606363321737, 1.1299145374598036, 1, 11, 1811, 410653)),
#     ((0.01, 4, 11, 8, 8), (143, 0, 0.00794606363321737, 1.1252162782666493, 1, 6, 1841, 416824)),
#     ((0.01, 4, 11, 11, 4), (0, 0, 0.009773524781915288, 1.226164619327706, 1, 56, 997, 261755)),
#     ((0.01, 4, 11, 11, 5), (0, 0, 0.009475947422650233, 1.2030354146291597, 1, 47, 1012, 265040)),
#     ((0.01, 4, 11, 11, 6), (0, 0, 0.009475947422650233, 1.1914049988718223, 1, 47, 844, 268496)),
#     ((0.01, 4, 11, 12, 5), (0, 0, 0.009018450247308074, 1.1902103689532915, 1, 40, 1030, 268878)),
#     ((0.01, 4, 11, 12, 6), (0, 0, 0.009018450247308074, 1.1820080968840652, 1, 38, 858, 272514)),
#     ((0.01, 4, 11, 12, 7), (0, 0, 0.008550528229663179, 1.1793803909323335, 1, 38, 870, 276150)),
#     ((0.01, 4, 11, 13, 6), (0, 0, 0.008588788169726858, 1.1764456814614366, 1, 33, 872, 276532)),
#     ((0.01, 4, 11, 13, 7), (0, 0, 0.008550528229663179, 1.1770092437932587, 1, 34, 884, 280168)),
#     ((0.01, 4, 11, 13, 9), (0, 0, 0.008550528229663179, 1.1715897456039561, 1, 32, 908, 287840)),
#     ((0.01, 4, 11, 13, 11), (0, 0, 0.009372866688597714, 1.1660804051310916, 1, 28, 932, 295124)),
#     ((0.01, 4, 11, 13, 12), (0, 0, 0.009010855143516405, 1.1677503573950474, 1, 33, 944, 298348)),
#     ((0.01, 4, 11, 13, 13), (0, 0, 0.008550528229663179, 1.1632295473995884, 1, 29, 956, 301984)),
#     ((0.01, 4, 11, 13, 15), (0, 0, 0.008550528229663179, 1.1627949643873559, 1, 29, 980, 309256)),
#     ((0.01, 4, 11, 13, 18), (0, 0, 0.008550528229663179, 1.1621793338324948, 1, 28, 1016, 320391)),
#     ((0.01, 4, 11, 14, 7), (0, 0, 0.008588788169726858, 1.1754789495094047, 1, 33, 898, 284186)),
#     ((0.01, 4, 11, 14, 9), (0, 0, 0.009372866688597714, 1.1750593795985431, 1, 31, 922, 291458)),
#     ((0.01, 4, 11, 14, 10), (0, 0, 0.009372866688597714, 1.1728402402107123, 1, 30, 934, 295094)),
#     ((0.01, 4, 11, 14, 13), (0, 0, 0.009010855143516405, 1.1655271617242409, 1, 28, 970, 306002)),
#     ((0.01, 4, 11, 14, 14), (0, 0, 0.008550528229663179, 1.1636133241707733, 1, 28, 982, 309857)),
#     ((0.01, 4, 11, 15, 11), (0, 0, 0.009372866688597714, 1.1665106357091062, 1, 27, 960, 302962)),
#     ((0.01, 4, 11, 15, 12), (0, 0, 0.009010855143516405, 1.165185020643653, 1, 27, 972, 306167)),
#     ((0.01, 4, 11, 15, 14), (0, 0, 0.008550528229663179, 1.1630376230119086, 1, 26, 996, 313433)),
#     ((0.01, 4, 11, 15, 16), (0, 0, 0.008550528229663179, 1.1629297502783433, 1, 25, 1020, 320699)),
#     ((0.01, 4, 12, 12, 5), (0, 0, 0.009836157895187103, 1.1920211734361454, 1, 44, 860, 272413)),
#     ((0.01, 4, 12, 13, 6), (0, 0, 0.008550528229663179, 1.1780172823617332, 1, 33, 886, 280614)),
#     ((0.01, 4, 12, 13, 7), (0, 0, 0.009773524781915288, 1.1773669265254878, 1, 32, 898, 284244)),
#     ((0.01, 4, 12, 13, 9), (0, 0, 0.009010855143516405, 1.1702006456035094, 1, 32, 922, 290895)),
#     ((0.01, 4, 12, 14, 9), (0, 0, 0.009372866688597714, 1.1715402515685847, 1, 28, 936, 294893)),
#     ((0.01, 4, 12, 14, 12), (0, 0, 0.009010855143516405, 1.1693917719852942, 1, 28, 972, 305972)),
#     ((0.01, 4, 13, 12, 7), (0, 0, 0.008550528229663179, 1.175628031090037, 1, 35, 898, 284894)),
#     ((0.01, 4, 13, 13, 6), (0, 0, 0.008550528229663179, 1.1753095011825063, 1, 32, 900, 285275)),
#     ((0.01, 4, 13, 13, 8), (0, 0, 0.008550528229663179, 1.17729897853438, 1, 31, 924, 292744)),
#     ((0.01, 4, 13, 14, 7), (0, 0, 0.008550528229663179, 1.1733689086985606, 1, 31, 926, 292922)),
#     ((0.01, 4, 13, 14, 8), (0, 0, 0.008550528229663179, 1.1776317327617825, 1, 30, 938, 296762)),
#     ((0.01, 4, 13, 14, 9), (0, 0, 0.008550528229663179, 1.1739019070615544, 1, 28, 950, 300188)),
#     ((0.01, 4, 14, 8, 8), (149, 0, 0.00794606363321737, 1.1258820998702361, 1, 5, 1934, 445630)),
#     ((0.01, 5, 10, 11, 3), (1, 0, 0.009178976435896935, 1.1719939502835668, 1, 38, 1196, 277550)),
#     ((0.01, 5, 10, 12, 3), (1, 0, 0.009178976435896935, 1.1704545563475401, 1, 37, 1218, 281655)),
#     ((0.01, 5, 10, 13, 3), (0, 0, 0.009178976435896935, 1.1704545563475401, 1, 37, 1051, 285752)),
#     ((0.01, 5, 12, 14, 6), (0, 0, 0.008550528229663179, 1.1766473500570975, 1, 30, 940, 296728)),
#     ((0.01, 5, 13, 8, 7), (147, 0, 0.00794606363321737, 1.1258820998702361, 1, 5, 1972, 451062)),
#     ((0.01, 5, 13, 13, 6), (0, 0, 0.008550528229663179, 1.1745551994327317, 1, 30, 940, 297389)),
#     ((0.01, 5, 14, 13, 6), (0, 0, 0.008550528229663179, 1.171259039852144, 1, 30, 954, 302276)),
# ]
# final frontier for babylonian:
# [
#     ((0.01, 3, 7, 8, 10), (166, 0, 0.008550528229663179, 1.1637618974263193, 1, 36, 1698, 413452)),
#     ((0.01, 3, 7, 8, 11), (166, 0, 0.008550528229663179, 1.1525767064861412, 1, 25, 1708, 415422)),
#     ((0.01, 3, 7, 8, 12), (166, 0, 0.00856143725215519, 1.1522790613429763, 1, 20, 1718, 417388)),
#     ((0.01, 3, 7, 8, 13), (166, 0, 0.00856143725215519, 1.1508942582099326, 1, 17, 1728, 419682)),
#     ((0.01, 3, 7, 8, 14), (166, 0, 0.00856143725215519, 1.1506213307958757, 1, 16, 1738, 422143)),
#     ((0.01, 3, 8, 8, 10), (140, 0, 0.008550528229663179, 1.1501387818373434, 1, 32, 1719, 389639)),
#     ((0.01, 3, 8, 8, 11), (139, 0, 0.008550528229663179, 1.1389932973815138, 1, 20, 1729, 390303)),
#     ((0.01, 3, 8, 8, 12), (139, 0, 0.008550528229663179, 1.1363124121025838, 1, 14, 1739, 392242)),
#     ((0.01, 3, 8, 8, 13), (139, 0, 0.00794606363321737, 1.1353521082467488, 1, 11, 1749, 394011)),
#     ((0.01, 3, 8, 8, 14), (139, 0, 0.00794606363321737, 1.1350791808326919, 1, 10, 1759, 396114)),
#     ((0.01, 3, 8, 10, 9), (0, 0, 0.011059783418037927, 1.2826825491454596, 1, 74, 988, 258160)),
#     ((0.01, 3, 8, 10, 10), (0, 0, 0.011059783418037927, 1.3503857992051542, 1, 91, 993, 258132)),
#     ((0.01, 3, 8, 10, 11), (0, 0, 0.011059783418037927, 1.2698705317095749, 1, 71, 820, 258802)),
#     ((0.01, 3, 8, 10, 12), (0, 0, 0.011059783418037927, 1.2476076617708567, 1, 64, 824, 260182)),
#     ((0.01, 3, 8, 10, 13), (0, 0, 0.011059783418037927, 1.2385532349656194, 1, 61, 828, 261564)),
#     ((0.01, 3, 8, 10, 14), (0, 0, 0.011059783418037927, 1.23464757552119, 1, 60, 832, 262767)),
#     ((0.01, 3, 8, 11, 9), (0, 0, 0.010317283282404777, 1.2081478292954069, 1, 45, 834, 262958)),
#     ((0.01, 3, 8, 11, 13), (0, 0, 0.010317283282404777, 1.2014149218638148, 1, 49, 850, 267580)),
#     ((0.01, 3, 8, 11, 14), (0, 0, 0.010317283282404777, 1.201223867646374, 1, 48, 854, 268594)),
#     ((0.01, 3, 8, 12, 9), (0, 0, 0.009773524781915288, 1.195476715842955, 1, 45, 856, 269528)),
#     ((0.01, 3, 8, 12, 10), (0, 0, 0.009836157895187103, 1.189974183035924, 1, 40, 860, 270542)),
#     ((0.01, 3, 8, 13, 11), (0, 0, 0.010317283282404777, 1.190887083251261, 1, 36, 886, 277930)),
#     ((0.01, 3, 8, 13, 12), (0, 0, 0.009595376126073862, 1.1772560926720206, 1, 34, 890, 279328)),
#     ((0.01, 3, 8, 13, 14), (0, 0, 0.009595376126073862, 1.1763603898966242, 1, 31, 898, 281732)),
#     ((0.01, 3, 8, 14, 14), (0, 0, 0.009595376126073862, 1.1766931441240267, 1, 30, 920, 288114)),
#     ((0.01, 3, 9, 8, 11), (143, 0, 0.008550528229663179, 1.1303401704759486, 1, 14, 1750, 396882)),
#     ((0.01, 3, 9, 8, 12), (143, 0, 0.008550528229663179, 1.1274480853675306, 1, 8, 1760, 398830)),
#     ((0.01, 3, 9, 8, 13), (143, 0, 0.00794606363321737, 1.126155027284293, 1, 6, 1770, 400436)),
#     ((0.01, 3, 9, 8, 14), (143, 0, 0.00794606363321737, 1.1258820998702361, 1, 5, 1780, 402549)),
#     ((0.01, 3, 9, 12, 10), (0, 0, 0.009773524781915288, 1.1888758676545883, 1, 39, 870, 277027)),
#     ((0.01, 3, 9, 12, 11), (0, 0, 0.008550528229663179, 1.1893529364556823, 1, 40, 874, 278054)),
#     ((0.01, 3, 9, 12, 12), (0, 0, 0.008550528229663179, 1.1883841904674723, 1, 40, 878, 278886)),
#     ((0.01, 3, 9, 12, 13), (0, 0, 0.008550528229663179, 1.1844782760804191, 1, 38, 882, 280102)),
#     ((0.01, 3, 9, 12, 16), (0, 0, 0.008550528229663179, 1.183309799393138, 1, 37, 894, 283947)),
#     ((0.01, 3, 9, 13, 11), (0, 0, 0.008550528229663179, 1.1835855151168726, 1, 34, 896, 284334)),
#     ((0.01, 3, 9, 13, 12), (0, 0, 0.008550528229663179, 1.1789378892706182, 1, 34, 900, 285551)),
#     ((0.01, 3, 9, 13, 13), (0, 0, 0.008550528229663179, 1.1814833353361738, 1, 33, 904, 286768)),
#     ((0.01, 3, 9, 13, 14), (0, 0, 0.008550528229663179, 1.1780961228620044, 1, 32, 908, 288387)),
#     ((0.01, 3, 9, 13, 15), (0, 0, 0.008550528229663179, 1.1774804923071434, 1, 31, 912, 289606)),
#     ((0.01, 3, 9, 14, 13), (0, 0, 0.008550528229663179, 1.171281401171253, 1, 29, 926, 293240)),
#     ((0.01, 3, 10, 12, 11), (0, 0, 0.009010855143516405, 1.1875567211204758, 1, 40, 884, 279578)),
#     ((0.01, 3, 10, 12, 13), (0, 0, 0.008550528229663179, 1.1832824165220843, 1, 38, 892, 282388)),
#     ((0.01, 3, 11, 13, 12), (0, 0, 0.008550528229663179, 1.1753691630236853, 1, 33, 920, 291478)),
#     ((0.01, 3, 11, 14, 13), (0, 0, 0.008550528229663179, 1.16771267492432, 1, 28, 946, 298936)),
# ]
