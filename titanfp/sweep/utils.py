"""Sweep helpers."""

from ..fpbench import fpcparser

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
