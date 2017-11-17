import random
import timeit
import math

w = 11
p = 53

def big():
    return random.randint(0, 2**p) * (2 ** random.randint(0, 2**w))

def p2(x):
    twos = 0
    while x % 2 == 0 and x > 0:
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





if __name__ == '__main__':
    algos = [p2p, p2d, p2x]
    per_width = 3

    for w, p in [(5, 11), (8, 24), (11, 53), (15, 113), (19, 237)]:
        print('\nw={:d}, p={:d}\n'.format(w, p))
        for _ in range(per_width):
            x = big()
            reps = 100
            reference = p2(x)
            print('using: ~exp({:.2f})\n  {:d} repetitions'.format(math.log(x), reps))
            for a in algos:
                seconds = timeit.timeit(lambda: a(x), number=reps)
                print('algorithm {:4s} {:.4f}s'.format(a.__name__, seconds))
                result = a(x)
                print('  got: {}, expected: {}, equal: {}'.format(result, reference, result == reference))
