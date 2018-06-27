"""Evaluation context information, shared across arithmetics."""


class EvalCtx(object):
    
    def __init__(self):
        self.bindings = {}

    def let(self, bindings):
        self.bindings.update(bindings)
        return self


# IEEE 754-like
class IEEECtx(EvalCtx):
    
    def __init__(self, w = 11, p = 53, props = None):
        super().__init__()
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
            elif prec in {'binary128', 'float128', 'quadruple'}:
                w = 15
                p = 113
            
        self.w = w
        self.p = p
        self.emax = (1 << (self.w - 1)) - 1
        self.emin = 1 - self.emax
        self.n = self.emin - self.p
    
    def clone(self):
        copy = IEEECtx(w=self.w, p=self.p)
        return copy.let(self.bindings)


# John Gustafson's Posits
class PositCtx(EvalCtx):

    def __init__(self, es = 4, nbits = 64, props = None):
        super().__init__()
        if props:
            prec = str(props.get('precision', '')).lower()
            if prec in {'binary64', 'float64', 'double'}:
                es = 4
                nbits = 64
            elif prec in {'binary32', 'float32', 'single'}:
                es = 3
                nbits = 32
            elif prec in {'binary16', 'float16', 'half'}:
                es = 1
                nbits = 16
            elif prec in {'binary128', 'float128', 'quadruple'}:
                es = 7
                nbits = 128

        self.es = es
        self.nbits = nbits
        self.u = 2 ** es

    def clone(self):
        copy = PositCtx(es=self.es, nbits=self.nbits)
        return copy.let(self.bindings)
