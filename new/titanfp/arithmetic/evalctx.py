"""Evaluation context information, shared across arithmetics."""

from ..titanic.ops import RM
from ..titanic import gmpmath


binary8_synonyms = {'binary8', 'posit8', 'posit8_t', 'quad'}
binary16_synonyms = {'binary16', 'float16', 'float16_t', 'posit16', 'posit16_t', 'half'}
binary32_synonyms = {'binary32', 'float32', 'float32_t', 'posit32', 'posit32_t', 'single', 'float'}
binary64_synonyms = {'binary64', 'float64', 'float64_t', 'posit64', 'posit64_t', 'double'}
binary128_synonyms = {'binary128', 'float128', 'float128_t', 'posit128', 'posit128_t', 'quadruple'}

RNE_synonyms = {'rne', 'nearesteven', 'roundnearesttiestoeven'}
RNA_synonyms = {'rna', 'nearestaway', 'roundnearesttiestoaway'}
RTP_synonyms = {'rtp', 'topositive', 'roundtowardpositive'}
RTN_synonyms = {'rtn', 'tonegative', 'roundtowardnegative'}
RTZ_synonyms = {'rtz', 'tozero', 'roundtowardzero'}
RAZ_synonyms = {'raz', 'awayzero', 'roundawayzero'}


class EvalCtx(object):

    def __init__(self):
        self.bindings = {}

    def let(self, bindings):
        self.bindings.update(bindings)
        return self


# IEEE 754-like
class IEEECtx(EvalCtx):

    def __init__(self, w = 11, p = 53, rm = RM.RNE, props = None):
        super().__init__()

        if props:
            prec = str(props.get('precision', '')).lower()
            if prec in binary16_synonyms:
                w = 5
                p = 11
            elif prec in binary32_synonyms:
                w = 8
                p = 24
            elif prec in binary64_synonyms:
                w = 11
                p = 53
            elif prec in binary128_synonyms:
                w = 15
                p = 113
            elif prec:
                raise ValueError('IEEECtx: unknown precision {}'.format(prec))

            rnd = str(props.get('round', '')).lower()
            if rnd in RNE_synonyms:
                rm = RM.RNE
            elif rnd in RNA_synonyms:
                rm = RM.RNA
            elif rnd in RTP_synonyms:
                rm = RM.RTP
            elif rnd in RTN_synonyms:
                rm = RM.RTN
            elif rnd in RTZ_synonyms:
                rm = RM.RTZ
            elif rnd:
                raise ValueError('IEEECtx: unknown rounding mode {}'.format(rnd))

        self.rm = rm
        self.w = w
        self.p = p
        self.emax = (1 << (self.w - 1)) - 1
        self.emin = 1 - self.emax
        self.n = self.emin - self.p
        self.fbound = gmpmath.ieee_fbound(self.w, self.p)

    def clone(self):
        copy = IEEECtx(w=self.w, p=self.p)
        return copy.let(self.bindings)


# John Gustafson's Posits
class PositCtx(EvalCtx):

    def __init__(self, es = 4, nbits = 64, props = None):
        super().__init__()

        if props:
            prec = str(props.get('precision', '')).lower()
            if prec in binary8_synonyms:
                es = 0
                nbits = 8
            elif prec in binary16_synonyms:
                es = 1
                nbits = 16
            elif prec in binary32_synonyms:
                es = 3
                nbits = 32
            elif prec in binary64_synonyms:
                es = 4
                nbits = 64
            elif prec in binary128_synonyms:
                es = 7
                nbits = 128
            elif prec:
                raise ValueError('PositCtx: unknown precision {}'.format(prec))

        self.es = es
        self.nbits = nbits
        self.u = 1 << es
        self.emax = 1 << (self.nbits - 2)
        self.emin = -self.emax

    def clone(self):
        copy = PositCtx(es=self.es, nbits=self.nbits)
        return copy.let(self.bindings)
