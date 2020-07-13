"""Sweep helpers."""

import random
import json

from ..fpbench import fpcparser
from ..arithmetic import ieee754, posit


# convenient rounding contexts
bf16 = ieee754.ieee_ctx(8, 16)
f32 = ieee754.ieee_ctx(8, 32)
f64 = ieee754.ieee_ctx(11, 64)
f4k = ieee754.ieee_ctx(20, 4096)
posit16_1 = posit.posit_ctx(1, 16)
posit32_1 = posit.posit_ctx(1, 32)


def linear_ulps(x, y):
    smaller_n = min(x.n, y.n)
    x_offset = x.n - smaller_n
    y_offset = y.n - smaller_n

    x_m = x.m << x_offset
    y_m = y.m << y_offset

    return x_m - y_m

def load_cores(interpreter, cores, analyses=None):
    if isinstance(cores, str):
        cores = fpcparser.compile(cores)

    main = cores[-1]
    for core in cores:
        interpreter.register_function(core)
        if core.ident and core.ident.lower() == 'main':
            main = core

    if analyses:
        interpreter.analyses = analyses

    return main

def neighborhood(lo, hi, near):
    def neighbors(x):
        for n in range(x-near, x+near+1):
            if lo <= n <= hi and n != x:
                yield n
    return neighbors

def static_neighborhood(v):
    def init_static():
        return v
    def neighbor_static(x):
        yield v
    return init_static, neighbor_static

def integer_neighborhood(lo, hi, near):
    def init_random():
        return random.randint(lo, hi)
    return init_random, neighborhood(lo, hi, near)


def jsonlog(fname, gens, cfgs, frontier, settings=None):
    data = {'generations' : gens,
            'configs' : list(cfgs),
            'frontier' : frontier}
    if settings:
        data['settings'] = str(settings)

    with open(fname, 'wt') as f:
        json.dump(data, f)
        print(file=f, flush=True)
