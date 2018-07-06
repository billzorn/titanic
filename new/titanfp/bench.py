import random

from .fpbench import fpcparser
from .arithmetic import ieee754
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


    return ulps, agreement, one_ulp_agreement, zero_ulp_agreement


ctx128 = evalctx.IEEECtx(w=32, p=128)

def bench_core(core, hi_args, lo_args, ctx):
    hi_result = ieee754.interpret(core, hi_args, ctx=ctx)
    lo_result = ieee754.interpret(core, lo_args, ctx=ctx)
    return bits_agreement(hi_result, lo_result)

def sweep_core_1arg(core, erange, prange, benches, nbits, ctx):
    print(core)
    print('running with {} total bits'.format(nbits))
    
    for e in erange:
        for p in prange:
            for _ in range(benches):
                hi_arg, lo_arg = gen_input(e, p, nbits)
                ulps, agreement, one_ulp_agreement, zero_ulp_agreement = bench_core(core, [hi_arg], [lo_arg], ctx)

                print(lo_arg.n, one_ulp_agreement, zero_ulp_agreement)
                if one_ulp_agreement <= 2 or zero_ulp_agreement <= 1:
                    print(hi_arg)
                    print(lo_arg)



benchmarks = {
    'add' : '(FPCore (x y) (+ x y))',
    'sin' : '(FPCore (x) (sin x))',
    'floor' : '(FPCore (x) (floor x))',
}

cores = { k : fpcparser.compile(v)[0] for k, v in benchmarks.items() }
