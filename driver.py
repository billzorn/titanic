import os
import time
import multiprocessing
import z3

import ulps
import benchmarks

report_stats = [True]
default_domain = {
    'min_input' : None,
    'min_open'  : True,
    'max_input' : None,
    'max_open'  : True,
    # 'fplo'   : ulps.default_fplo,
    # 'fphi'   : ulps.default_fphi,
    # 'lotohi' : ulps.fp_up,
    # 'hitolo' : ulps.fp_down,
}
special_queries = {
    ulps.with_NaN_r       : 'r=NaN',
    ulps.with_NaN_c       : 'c=NaN',
    ulps.with_Inf_r_pos   : 'r=+Inf',
    ulps.with_Inf_r_neg   : 'r=-Inf',
    ulps.with_Inf_c_pos   : 'c=+Inf',
    ulps.with_Inf_c_neg   : 'c=-Inf',
    ulps.with_zeros_r_pos : 'r=+0.0, c=-0.0',
    ulps.with_zeros_r_neg : 'r=-0.0, c=+0.0',
}
queries_special = {special_queries[k] : k for k in special_queries}
    
def run_special(descr, query, expr, nargs,
                min_input = None, min_open = True, max_input = None, max_open = True,
                fplo = ulps.default_fplo, fphi = ulps.default_fphi, lotohi = ulps.fp_up, hitolo = ulps.fp_down):
    executable_query = queries_special[query]
    s = executable_query(expr, nargs,
                         min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open,
                         fplo=fplo, fphi=fphi, lotohi=lotohi, hitolo=hitolo)
    start = time.time()
    status = ulps.run_query(s, descr)
    end = time.time()
    m = ulps.serialize_model(s)
    if report_stats[0]:
        stats = ulps.serialize_stats(s)
    else:
        stats = None
    return descr, m, nargs, stats, end - start

def run_ulps(descr, expr, nargs, min_ulps,
                min_input = None, min_open = True, max_input = None, max_open = True,
                fplo = ulps.default_fplo, fphi = ulps.default_fphi, lotohi = ulps.fp_up, hitolo = ulps.fp_down):
    s = ulps.with_ulps(expr, nargs, min_ulps,
              min_input=min_input, min_open=min_open, max_input=max_input, max_open=max_open,
              fplo=fplo, fphi=fphi, lotohi=lotohi, hitolo=hitolo)
    start = time.time()
    status = ulps.run_query(s, descr)
    end = time.time()
    m = ulps.serialize_model(s)
    if report_stats[0]:
        stats = ulps.serialize_stats(s)
    else:
        stats = None
    return descr, m, nargs, stats, end - start

def report_callback(arg):
    ulps.report_query(*arg)
    
def run_pool(expr, nargs, ulps, domain = default_domain, workers=os.cpu_count()):
    with multiprocessing.Pool(processes=workers) as pool:
        results = []
        # for i in range(4):
        #     results.append(pool.apply_async(f, (i, 30,), {}, cb))
        for query in queries_special:
            results.append(pool.apply_async(run_special,
                                            (query, query, expr, nargs,),
                                            domain,
                                            report_callback))
        for n in ulps:
            results.append(pool.apply_async(run_ulps,
                                            ('>= {:d} ulps'.format(n), expr, nargs, n,),
                                            domain,
                                            report_callback))
        for r in results:
            r.wait()
            # print(r.get())
        print('All queries finished. Done.')

if __name__ == '__main__':
    run_pool(benchmarks.fpc_rosa, 1, [2,4,8,16])
