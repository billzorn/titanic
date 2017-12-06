import random
import timeit
import math

w = 11
p = 53

def v(_):
    return

def big():
    #return random.randint(0, 2**p) * (2 ** random.randint(0, 2**w))
    #return random.randint(0, 2 ** (2**w))
    return 2 ** random.randint(0, (2**w))

def p2(x):
    twos = 0
    while x % 2 == 0 and x > 0:
        x = x >> 1
        twos += 1
    return twos, x

def p2a(x):
    twos = 0
    while x & 1 == 0 and x > 0:
        x = x >> 1
        twos += 1
    return twos, x

def p2p(x):
    twos = 0
    f = 2
    f2 = 1
    while x % f == 0 and x > 0:
        x = x >> f2
        twos += f2
        f = f << 1
        f2 += 1
    while f > 2:
        f = f >> 1
        f2 -= 1
        while x % f == 0 and x > 0:
            x = x >> f2
            twos += f2
    return twos, x

def p2d(x):
    twos = 0
    f = 2
    f2 = 1
    while x % f == 0 and x > 0:
        x = x >> f2
        twos += f2
        f = f << 1
        f2 += 1
    while f > 2:
        while x % f != 0 and f > 2:
            f = f >> 1
            f2 -= 1
        while x % f == 0 and x > 0:
            x = x >> f2
            twos += f2
    return twos, x

def p2x(x):
    twos = 0
    f2 = 1
    fm = 1
    while x & fm == 0 and x > 0:
        x = x >> f2
        twos += f2
        f2 += 1
        fm = (fm << 1) + 1
    while fm > 1:
        while x & fm != 0 and fm > 1:
            fm = fm >> 1
            f2 -= 1
        while x & fm == 0 and x > 0:
            x = x >> f2
            twos += f2
    return twos, x

def p2z(x):
    twos = 0
    f2 = 1
    fm = 1
    while x & fm == 0 and x > 0:
        x = x >> f2
        twos += f2
        fm = fm + (fm << f2)
        f2 += f2
    while fm > 1:
        while x & fm != 0 and fm > 1:
            fm = fm >> 1
            f2 -= 1
        while x & fm == 0 and x > 0:
            x = x >> f2
            twos += f2
    return twos, x

def p2xs(x):
    twos = 0
    f2 = 1
    fm = 1
    while x & fm == 0 and x > 0:
        x = x >> f2
        twos += f2
        f2 += 1
        fm = (fm << 1) | 1
    # for f2 in range(f2-1, 1, -1):
    while f2 > 1:
        f2 -= 1
        fm = fm >> 1
        if x & fm == 0:
            x = x >> f2
            twos += f2
    return twos, x

# I am fairly confident this is the fastest pure python CTZ implementation
def p2xr(x):
    twos = 0
    f2 = 1
    fm = 1
    while x & fm == 0 and x > 0:
        x = x >> f2
        twos += f2
        f2 += 1
        fm = (fm << 1) | 1
    for f2 in range(f2-1, 1, -1):
    # while f2 > 1:
    #     f2 -= 1
        fm = fm >> 1
        if x & fm == 0:
            x = x >> f2
            twos += f2
    return twos, x


# Get the bits of a rational number 2**e * (p/q)'s binary expansion.
# This assumes that that p and q have the same number of bits. If they don't, modify e and scale q accordingly.
# It's also kind of assumed that p and q don't start with any trailing zeros, though we may add some to q
# in order to match the size of p.
# Big Idea: compare, subtract p/q - (q/q = 1) to get this bit (to the left of the binary point) if p/q >= 1,
# then p << 1 to change which bit we're looking at / where the current binary point is.
# This may eventually produce a cycle: we could detect that to improve performance.





if __name__ == '__main__':
    # algos = [v, p2, p2p, p2d, p2x, p2z, p2xs, p2xr]
    algos = [v, p2a, p2x, p2xs, p2xr]
    reps = 10000
    per_width = 3
    easy = [(5, 11), (8, 24), (11, 53)]
    hard = [(5, 11), (8, 24), (11, 53), (15, 113), (19, 237)]
    worst = [(12, 1024)]

    for w, p in worst:
        print('\nw={:d}, p={:d}\n'.format(w, p))
        for _ in range(per_width):
            x = big()
            reference = p2(x)
            print('using: ~exp({:.2f})\n  {:d} repetitions'.format(math.log(x), reps))
            for a in algos:
                seconds = timeit.timeit(lambda: a(x), number=reps)
                #seconds = timeit.timeit(a.__name__ + '(x)', number=reps)
                print('algorithm {:4s} {:.4f}s'.format(a.__name__, seconds))
                result = a(x)
                print('  got: {}, expected: {}, equal: {}'.format(result, reference, result == reference))
