"""Sweep helpers."""

import random
import json

from ..fpbench import fpcparser
from ..arithmetic import ieee754, posit, evalctx


# convenient rounding contexts
f8 = ieee754.ieee_ctx(3, 8)
f16 = ieee754.ieee_ctx(5, 16)
bf16 = ieee754.ieee_ctx(8, 16)
f32 = ieee754.ieee_ctx(8, 32)
f64 = ieee754.ieee_ctx(11, 64)

f4k = ieee754.ieee_ctx(20, 4096)

posit8_0 = posit.posit_ctx(0, 8)
posit16_1 = posit.posit_ctx(1, 16)
posit16_2 = posit.posit_ctx(2, 16)
posit32_2 = posit.posit_ctx(2, 32)
posit64_3 = posit.posit_ctx(3, 64)

float_basecase = (f8, f16, bf16, f32)
posit_basecase = (posit8_0, posit16_1, posit16_2, posit32_2)

def describe_ctx(ctx):
    if isinstance(ctx, evalctx.IEEECtx):
        if ctx.es == 3 and ctx.nbits == 8:
            return 'float8'
        elif ctx.es == 5 and ctx.nbits == 16:
            return 'float16'
        elif ctx.es == 8 and ctx.nbits == 16:
            return 'bfloat16'
        elif ctx.es == 8 and ctx.nbits == 32:
            return 'float32'
        elif ctx.es == 11 and ctx.nbits == 32:
            return 'float64'
        else:
            return f'(float {ctx.es!s} {ctx.nbits!s})'

    elif isinstance(ctx, evalctx.PositCtx):
        if ctx.es == 0 and ctx.nbits == 8:
            return 'posit8_0'
        elif ctx.es == 1 and ctx.nbits == 16:
            return 'posit16_1'
        elif ctx.es == 2 and ctx.nbits == 16:
            return 'posit16_2'
        elif ctx.es == 2 and ctx.nbits == 32:
            return 'posit32_2'
        elif ctx.es == 3 and ctx.nbits == 64:
            return 'posit64_3'
        else:
            return f'(posit {ctx.es!s} {ctx.nbits!s})'

    else:
        return ctx.propstr()

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
            if lo <= n <= hi:
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
