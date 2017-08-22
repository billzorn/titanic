from bv import BV
from real import FReal
import core
import conv

import numpy

def closer(n, w, p, rm, thresh, seed=None):

    rand = numpy.random.RandomState(seed % (2**31 - 1))

    umax = ((2 ** w) - 1) * (2 ** (p - 1))


    for x in range(n):
        sign = rand.randint(0, 2)
        i = rand.randint(0, umax-1) * sign

        offset = rand.randint(0, 2)
        if offset < 1:
            offset = -1


    # S1, E1, T1 = core.real_to_implicit(FReal('1e32'), w, p, rm)
    # i1 = core.implicit_to_ordinal(S1, E1, T1)
    # span = 1000

    # for i in range(i1 - span, i1 + span):
    #     offset = 1


        i2 = i + offset

        S, E, T = core.ordinal_to_implicit(i, w, p)
        S2, E2, T2 = core.ordinal_to_implicit(i2, w, p)

        R = core.implicit_to_real(S, E, T)
        R2 = core.implicit_to_real(S2, E2, T2)

        if offset > 0:
            i_below = i
            i_above = i2
            midpoint = R + ((R2 - R) / 2)
        else:
            i_below = i2
            i_above = i
            midpoint = R2 + ((R - R2) / 2)

        Sm, Em, Tm = core.ieee_round_to_implicit(midpoint, i_below, i_above, w, p, rm)

        # lower, lower_inclusive, upper, upper_inclusive = conv.implicit_to_rounding_envelope(Sm, Em, Tm, rm)

        prec, lowest, midlo, midhi, highest, e = conv.shortest_dec(midpoint, Sm, Em, Tm, rm, round_correctly=True)

        if prec >= thresh:
            print(i_below, ' >< ', i_above)
            print(conv.real_to_string(midpoint, exact=True))
            print(conv.pow10_to_str(highest, e))
            print(conv.pow10_to_str(lowest, e))
            print(' ', prec)
            print()

if __name__ == '__main__':
    import sys
    import time
    import multiprocessing

    ncores = int(sys.argv[1])
    w = int(sys.argv[2])
    p = int(sys.argv[3])
    rm = sys.argv[4]
    thresh = int(sys.argv[5])

    iters = 1000

    with multiprocessing.Pool(ncores, maxtasksperchild=10) as the_pool:

        seed = int.from_bytes(numpy.float64(time.time()).tobytes(), sys.byteorder)

        while True:
            results = [None] * ncores
            for i in range(ncores):
                seed += 1
                results[i] = the_pool.apply_async(closer, (iters, w, p, rm, thresh, seed,))
            for result in results:
                result.wait()
