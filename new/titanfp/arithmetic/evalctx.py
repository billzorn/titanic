"""Evaluation context information, shared across arithmetics."""


class EvalCtx(object):
    
    def __init__(self, w = 11, p = 53, props = None):
        # IEEE 754-like
        if props:
            prec = str(props.get('precision', '')).lower()
            if prec in {'binary64', 'float64', 'double'}:
                w = 11
                p = 53
            elif prec in {'binary32', 'float32', 'single'}:
                w = 8
                p = 24
            elif prec in {'binary16', 'float16', 'half'}:
                w = 5
                p = 11
            
        self.w = w
        self.p = p
        self.emax = (1 << (self.w - 1)) - 1
        self.emin = 1 - self.emax
        self.n = self.emin - self.p
            
        # variables and stuff
        self.bindings = {}

    def clone(self):
        copy = EvalCtx(w=self.w, p=self.p)
        copy.bindings.update(self.bindings)
        return copy

    def let(self, bindings):
        self.bindings.update(bindings)
