import random
import multiprocessing

from .fpbench import fpcparser
from .arithmetic import ieee754
from .arithmetic import optimistic
from .arithmetic import evalctx
from .titanic import sinking


def gen_input(e, p, nbits, negative=False):
    # p <= nbits

    significand = random.randint(0, (1 << (nbits - 1)) - 1) | (1 << (nbits - 1))

    hi = sinking.Sink(negative=negative, c=significand, exp=e - nbits + 1)
    lo = hi.round_m(p)

    return hi, lo


def linear_ulps(x, y):
    smaller_n = min(x.n, y.n)
    x_offset = x.n - smaller_n
    y_offset = y.n - smaller_n

    x_c = x.c << x_offset
    y_c = y.c << y_offset

    return x_c - y_c


def bits_agreement(hi, lo):
    ulps = linear_ulps(hi, lo)
    floorlog2_ulps = max(abs(ulps).bit_length() - 1, 0)

    # this should be an upper bound on the number of bits
    # that are the same
    lo_offset = lo.n - min(hi.n, lo.n)
    agreement = lo.p + lo_offset - floorlog2_ulps

    hi_exact = sinking.Sink(hi, inexact=False, rc=0)
    lo_exact = sinking.Sink(lo, inexact=False, rc=0)
    one_ulp_agreement = None
    zero_ulp_agreement = None
    for p in range(agreement, -1, -1):
        hi_rounded = hi_exact.round_m(p)
        lo_rounded = lo_exact.round_m(p)
        rounded_ulps = linear_ulps(hi_rounded, lo_rounded)
        if one_ulp_agreement is None and abs(rounded_ulps) <= 1:
            one_ulp_agreement = p
        if zero_ulp_agreement is None and rounded_ulps == 0:
            zero_ulp_agreement = p

        if one_ulp_agreement is not None and zero_ulp_agreement is not None:
            break

    if one_ulp_agreement == None:
        one_ulp_agreement = 0
    if zero_ulp_agreement == None:
        zero_ulp_agreement = 0

    return ulps, agreement, one_ulp_agreement, zero_ulp_agreement


ctx128 = evalctx.IEEECtx(w=32, p=128)
ctx64 = evalctx.IEEECtx(w=11, p=64)

sweep_verbose = True
rejections = 10

def bench_core(core, hi_args, lo_args, ctx):
    hi_result = ieee754.interpret(core, hi_args, ctx=ctx)
    lo_result = ieee754.interpret(core, lo_args, ctx=ctx)
    return bits_agreement(hi_result, lo_result)

def sweep_core_1arg(core, erange, prange, benches, nbits, ctx):
    print(core)
    print('running with {} total bits'.format(nbits))
    print(flush=True)

    records = []

    for e in erange:
        for p in prange:
            for _ in range(benches):
                hi_arg, lo_arg = gen_input(e, p, nbits)

                if core.pre:
                    for _ in range(rejections):
                        if (ieee754.interpret_pre(core, [hi_arg], ctx) and
                            ieee754.interpret_pre(core, [lo_arg], ctx)):
                            break
                        hi_arg, lo_arg = gen_input(e1, p1, nbits)

                    if not (ieee754.interpret_pre(core, [hi_arg], ctx) and
                            ieee754.interpret_pre(core, [lo_arg], ctx)):
                        print('failed to meet precondition')
                        continue

                ulps, agreement, one_ulp_agreement, zero_ulp_agreement = bench_core(core, [hi_arg], [lo_arg], ctx)
                sunk = optimistic.interpret(core, [lo_arg], ctx)

                if sunk.inexact:
                    records.append([sunk.p, one_ulp_agreement])

                if one_ulp_agreement < sunk.p - 1 and sweep_verbose:
                                print('??? expected ~{} bits of precision, got {} / {} / {}'.format(
                                    sunk.p, agreement, one_ulp_agreement, zero_ulp_agreement))
                                print(hi_arg, lo_arg)
                                print(sunk, ieee754.interpret(core, [hi_arg], ctx))
                                print()
    return records


def sweep_core_1arg_multi(core, erange, prange, benches, nbits, ctx, nprocs = None):
    if nprocs == None:
        nprocs = max(multiprocessing.cpu_count() // 2, 1)

    print(core)
    print('running with {} total bits, {} processes'.format(nbits, nprocs))
    print(flush=True)

    pool = multiprocessing.Pool(processes = nprocs)
    arg_iter = ([core, e, prange, benches, nbits, ctx] for e in erange)

    all_results = []
    for results in pool.starmap(sweep_core_1arg_inner, arg_iter, max(len(erange) // nprocs, 1)):
        if results is not None:
            all_results += results

    pool.close()
    pool.join()

    return all_results

def sweep_core_1arg_inner(core, e, prange, benches, nbits, ctx):
    records = []

    for p in prange:
        for _ in range(benches):
            hi_arg, lo_arg = gen_input(e, p, nbits)

            if core.pre:
                for _ in range(rejections):
                    if (ieee754.interpret_pre(core, [hi_arg], ctx) and
                        ieee754.interpret_pre(core, [lo_arg], ctx)):
                        break
                    hi_arg, lo_arg = gen_input(e1, p1, nbits)

                if not (ieee754.interpret_pre(core, [hi_arg], ctx) and
                        ieee754.interpret_pre(core, [lo_arg], ctx)):
                    print('failed to meet precondition')
                    continue

            ulps, agreement, one_ulp_agreement, zero_ulp_agreement = bench_core(core, [hi_arg], [lo_arg], ctx)
            sunk = optimistic.interpret(core, [lo_arg], ctx)

            if sunk.inexact:
                records.append([sunk.p, one_ulp_agreement])

    return records


def sweep_core_2arg(core, erange, prange, benches, nbits, ctx):
    print(core)
    print('running with {} total bits'.format(nbits))
    print(flush=True)

    records = []

    for e1 in erange:
        for e2 in erange:
            for p1 in prange:
                for p2 in prange:
                    for _ in range(benches):
                        hi_arg1, lo_arg1 = gen_input(e1, p1, nbits)
                        hi_arg2, lo_arg2 = gen_input(e2, p2, nbits)

                        if core.pre:
                            for _ in range(rejections):
                                if (ieee754.interpret_pre(core, [hi_arg1, hi_arg2], ctx) and
                                    ieee754.interpret_pre(core, [lo_arg1, lo_arg2], ctx)):
                                    break
                                hi_arg1, lo_arg1 = gen_input(e1, p1, nbits)
                                hi_arg2, lo_arg2 = gen_input(e2, p2, nbits)

                            if not (ieee754.interpret_pre(core, [hi_arg1, hi_arg2], ctx) and
                                    ieee754.interpret_pre(core, [lo_arg1, lo_arg2], ctx)):
                                print('failed to meet precondition')
                                continue

                        ulps, agreement, one_ulp_agreement, zero_ulp_agreement = bench_core(
                            core, [hi_arg1, hi_arg2], [lo_arg1, lo_arg2], ctx
                        )
                        sunk = optimistic.interpret(core, [lo_arg1, lo_arg2], ctx)

                        if sunk.inexact:
                            records.append([sunk.p, one_ulp_agreement])

                        if one_ulp_agreement < sunk.p - 1 and sweep_verbose:
                            print('??? expected ~{} bits of precision, got {} / {} / {}'.format(
                                sunk.p, agreement, one_ulp_agreement, zero_ulp_agreement))
                            print(hi_arg1, lo_arg1)
                            print(hi_arg2, lo_arg2)
                            print(sunk, ieee754.interpret(core, [hi_arg1, hi_arg2], ctx))
                            print()

    return records


def sweep_core_2arg_multi(core, erange, prange, benches, nbits, ctx, nprocs = None):
    if nprocs == None:
        nprocs = max(multiprocessing.cpu_count() // 2, 1)

    print(core)
    print('running with {} total bits, {} processes'.format(nbits, nprocs))
    print(flush=True)

    pool = multiprocessing.Pool(processes = nprocs)
    def arg_iter():
        for e1 in erange:
            for e2 in erange:
                yield [core, e1, e2, prange, benches, nbits, ctx]

    all_results = []
    for results in pool.starmap(sweep_core_2arg_inner, arg_iter(), max(len(erange) * len(erange) // nprocs, 1)):
        if results is not None:
            all_results += results

    pool.close()
    pool.join()

    return all_results

def sweep_core_2arg_inner(core, e1, e2, prange, benches, nbits, ctx):
    records = []

    for p1 in prange:
        for p2 in prange:
            for _ in range(benches):
                hi_arg1, lo_arg1 = gen_input(e1, p1, nbits)
                hi_arg2, lo_arg2 = gen_input(e2, p2, nbits)

                if core.pre:
                    for _ in range(rejections):
                        if (ieee754.interpret_pre(core, [hi_arg1, hi_arg2], ctx) and
                            ieee754.interpret_pre(core, [lo_arg1, lo_arg2], ctx)):
                            break
                        hi_arg1, lo_arg1 = gen_input(e1, p1, nbits)
                        hi_arg2, lo_arg2 = gen_input(e2, p2, nbits)

                    if not (ieee754.interpret_pre(core, [hi_arg1, hi_arg2], ctx) and
                            ieee754.interpret_pre(core, [lo_arg1, lo_arg2], ctx)):
                        print('failed to meet precondition')
                        continue

                ulps, agreement, one_ulp_agreement, zero_ulp_agreement = bench_core(
                    core, [hi_arg1, hi_arg2], [lo_arg1, lo_arg2], ctx
                )
                sunk = optimistic.interpret(core, [lo_arg1, lo_arg2], ctx)

                if sunk.inexact:
                    records.append([sunk.p, one_ulp_agreement])

    return records



benchmarks = {
    'add' : '(FPCore (x y) (+ x y))',
    'sub' : '(FPCore (x y) (- x y))',
    'mul' : '(FPCore (x y) (* x y))',
    'div' : '(FPCore (x y) (/ x y))',
    'sqrt' : '(FPCore (x) (sqrt x))',

    'floor' : '(FPCore (x) (floor x))',
    'fmod' : '(FPCore (x y) (fmod x y))',
    'sin' : '(FPCore (x) (sin x))',
    'pow' : '(FPCore (x y) (pow x y))',

    'quadratic' : """(FPCore (a b c)
 :name "NMSE p42, positive"
 :cite (hamming-1987 herbie-2015)
 :fpbench-domain textbook
 :pre (and (>= (* b b) (* 4 (* a c))) (!= a 0))
 (/ (+ (- b) (sqrt (- (* b b) (* 4 (* a c))))) (* 2 a)))
""",

    'ex3.1' : """(FPCore (x)
 :name "NMSE example 3.1"
 :cite (hamming-1987 herbie-2015)
 :fpbench-domain textbook
 :pre (>= x 0)
 (- (sqrt (+ x 1)) (sqrt x)))
""",

    'ex3.6' : """(FPCore (x)
 :name "NMSE example 3.6"
 :cite (hamming-1987 herbie-2015)
 :fpbench-domain textbook
 :pre (>= x 0)
 (- (/ 1 (sqrt x)) (/ 1 (sqrt (+ x 1)))))
""",

    'ex3.3.3' : """(FPCore (x)
 :name "NMSE problem 3.3.3"
 :cite (hamming-1987 herbie-2015)
 :fpbench-domain textbook
 :pre (!= x 0 1 -1)
 (+ (- (/ 1 (+ x 1)) (/ 2 x)) (/ 1 (- x 1))))
""",
}

cores = { k : fpcparser.compile(v)[0] for k, v in benchmarks.items() }
