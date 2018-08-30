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
    """Generic context for holding variable bindings and properties."""

    # these placeholders should never have anything put in them
    bindings = {}
    props = {}

    def __init__(self, bindings = None, props = None):
        if bindings:
            self.bindings = bindings.copy()
        else:
            self.bindings = {}

        if props:
            self.props = props.copy()
        else:
            self.props = {}

    def _import_fields(self, ctx):
        pass

    def __repr__(self):
        args = []
        if len(self.bindings) > 0:
            args.append('bindings=' + repr(self.bindings))
        if len(self.props) > 0:
            args.append('props=' + repr(self.props))
        return '{}({})'.format(type(self).__name__, ', '.join(args))

    def __str__(self):
        fields = ['    ' + str(k) + ': ' + str(v) for k, v in self.__dict__.items() if k not in {'bindings', 'props'}]
        bindings = ['    ' + str(k) + ': ' + str(v) for k, v in self.bindings.items()]
        props = ['    ' + str(k) + ': ' + str(v) for k, v in self.props.items()]
        if len(bindings) > 0:
            fields.append('  bindings:')
        if len(props) > 0:
            bindings.append('  props:')
        return '\n'.join([
            type(self).__name__ + ':',
            *fields,
            *bindings,
            *props
        ])

    def let(self, bindings = None, props = None):
        """Create a new context, updated with any provided bindings
        or properties.
        """
        cls = self.__class__
        newctx = cls.__new__(cls)

        # if bindings:
        #     newctx.bindings = self.bindings.copy()
        #     newctx.bindings.update(bindings)
        # else:
        #     newctx.bindings = self.bindings

        # if props:
        #     newctx.bindings = self.bindings.copy()
        #     newctx.props.update(props)
        # else:
        #     newctx.props = self.props

        # safer
        newctx.bindings = {}
        newctx.bindings.update(self.bindings)
        if bindings is not None:
            newctx.bindings.update(bindings)
        newctx.props = {}
        newctx.props.update(self.props)
        if props is not None:
            newctx.props.update(props)

        newctx._import_fields(self)
        return newctx


class IEEECtx(EvalCtx):
    """Context for IEEE 754-like arithmetic."""

    w = 11
    p = 53
    rm = RM.RNE
    emax = (1 << (w - 1)) - 1
    emin = 1 - emax
    n = emin - p
    fbound = gmpmath.ieee_fbound(w, p)

    def __init__(self, w = w, p = p, rm = rm, bindings = None, props = None):
        super().__init__(bindings=bindings, props=props)

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

    def _import_fields(self, ctx):
        self.rm = ctx.rm
        self.w = ctx.w
        self.p = ctx.p
        self.emax = ctx.emax
        self.emin = ctx.emin
        self.n = ctx.n
        self.fbound = ctx.fbound

    def __repr__(self):
        args = ['w=' + repr(self.w), 'p=' + repr(self.p), 'rm=' + str(self.rm)]
        if len(self.bindings) > 0:
            args.append('bindings=' + repr(self.bindings))
        if len(self.props) > 0:
            args.append('props=' + repr(self.props))
        return '{}({})'.format(type(self).__name__, ', '.join(args))


# John Gustafson's Posits
class PositCtx(EvalCtx):
    """Context for John Gustafson's posit arithmetic."""

    es = 4
    nbits = 64
    u = 1 << es
    emax = 1 << (nbits - 2)
    emin = -emax

    def __init__(self, es = es, nbits = nbits, bindings = None, props = None):
        super().__init__(bindings=bindings, props=props)

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

    def _import_fields(self, ctx):
        self.es = ctx.es
        self.nbits = ctx.nbits
        self.u = ctx.u
        self.emax = ctx.emax
        self.emin = ctx.emin

    def __repr__(self):
        args = ['es=' + repr(self.es), 'nbits=' + repr(self.nbits)]
        if len(self.bindings) > 0:
            args.append('bindings=' + repr(self.bindings))
        if len(self.props) > 0:
            args.append('props=' + repr(self.props))
        return '{}({})'.format(type(self).__name__, ', '.join(args))
