import core
import conv
import testgen
import fpcast as ast
import fpcsolver
FPSolver = fpcsolver.FPSolver
from real import FReal

sorts_8_12 = ((4, 4,), (4, 8,),)
sorts_8_16 = ((4, 4,), (5, 11,),)
sorts_16_20 = ((5, 11,), (5, 15,),)
sorts_16_24 = ((5, 11,), (6, 18,),)
sorts_16_32 = ((5, 11,), (8, 24,),)
sorts_32_64 = ((8, 24,), (11, 53,),)

main_sorts = sorts_8_12
other_sorts = [
    (sorts_8_12, 30),
    (sorts_8_16, 30),
    (sorts_16_20, 120),
    (sorts_16_24, 120),
    (sorts_16_32, 120),
    (sorts_32_64, 120),
]

def break_core_at_sorts(coreobj, sorts, timeout_ms):
    lo_sort, hi_sort = sorts
    solver = FPSolver(coreobj, lo_sort=lo_sort, hi_sort=hi_sort, timeout_ms=timeout_ms)
    w, p = lo_sort
    target_ulps = 2 ** (p - 1)
    reached_ulps = solver.binsearch_ulps(target_ulps)

    if reached_ulps is None:
        return 0, {}
    else:
        m = solver.get_model()
        args_dict = {arg : m[fpcsolver.arglo(i, arg)] for i, arg in enumerate(coreobj.args)}
        return reached_ulps, args_dict

def nbools(n):
    if n == 0:
        yield ()
    else:
        for bools in nbools(n-1):
            yield (True, *bools)
            yield (False, *bools)

def scale_fp(r_str, sort_lo, sort_hi):
    w_lo, p_lo = sort_lo
    w_hi, p_hi = sort_hi
    r = FReal(r_str)
    S, E, T = core.real_to_implicit(r, w_lo, p_lo, core.RNE)
    s, Re, Rf = core.implicit_to_relative(S, E, T)
    scaled_value = core.implicit_to_real(*core.relative_to_implicit(s, Re, Rf, w_hi, p_hi))
    prec = conv.bdb_round_trip_prec(p_hi)
    return conv.real_to_string(scaled_value, prec=prec, exact=False).lstrip(conv.approx_str)

def true_ulps(coreobj, args_dict, w = 8, p = 24):
    results = coreobj.e.apply_all(args_dict, (w, p), 'RNE')
    return ast.results_to_ulps(results, w, p)

def scale_breakage(coreobj, ulps, args_dict, w_lo, p_lo):
    w_hi, p_hi = 8, 24
    # we want all the bits on the right to be bad, ideally
    extra_bits = p_hi - p_lo
    want_ulps = ulps << extra_bits
    best_ulps = 0
    best_strat = None
    best_args = None
    best_results = None
    for strategy in nbools(len(coreobj.args)):
        strat_args = {}
        for i, arg in enumerate(coreobj.args):
            if strategy[i]:
                strat_args[arg] = scale_fp(args_dict[arg], (w_lo, p_lo), (w_hi, p_hi))
            else:
                strat_args[arg] = args_dict[arg]
        strat_ulps, rounded, expected = true_ulps(coreobj, strat_args, w_hi, p_hi)
        #print(strat_args, strat_ulps, rounded, expected)
        if strat_ulps > want_ulps:
            return strat_ulps, strategy, strat_args, (rounded, expected)
        elif strat_ulps > best_ulps:
            best_ulps = strat_ulps
            best_strat = strategy
            best_args = strat_args
            best_results = rounded, expected
    return best_ulps, best_strat, best_args, best_results

def test_core(coreobj):
    res_str = ['']
    def prnt(*args):
        #res_str[0] += ' '.join(str(arg) for arg in args) + '\n'
        print(*args)

    prnt('>>>args', coreobj.args)
    prnt('>>>core', coreobj.e)

    for sorts, timeout_s in other_sorts:
        try:
            prnt('>>>sort ', sorts)
            lo_sort, hi_sort = sorts
            w_lo, p_lo = lo_sort
            w_hi, p_hi = hi_sort
            sort_str = '{:d},{:d};{:d},{:d}'.format(w_lo, p_lo, w_hi, p_hi)

            ulps, args_dict = break_core_at_sorts(coreobj, sorts, timeout_ms=1000*timeout_s)
            prnt('>>>solved', sort_str, ulps, args_dict)
            if ulps > 0:
                scaled_ulps, strategy, scaled_args, scaled_results = scale_breakage(coreobj, ulps, args_dict, w_lo, p_lo)
                prnt('>>>strat', sort_str, '', strategy)
                prnt('>>>best', sort_str, ' ', scaled_ulps, scaled_args, scaled_results)
        except Exception as e:
            prnt(e)

    return res_str[0]

if __name__ == '__main__':
    import argparse
    from multiprocessing import Pool
    import random

    # for e in testgen.gen_up_to_depth(2, 1):
    #     coreobj = testgen.mkcore(e, 1)
    #     test_core(coreobj)

    # for coreobj in testgen.gen_from_file('allcores.fpcore'):
    #     test_core(coreobj)

    for i in range(2000):
        depth = random.choice(range(3, 6))
        nargs = random.choice(range(1, 4))
        coreobj = testgen.mkcore(testgen.gen_random_depth(depth, nargs), nargs)
        test_core(coreobj)
