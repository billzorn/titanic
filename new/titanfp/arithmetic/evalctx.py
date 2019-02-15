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

        self.props = {}
        if props:
            self._update_props(props)

    def _update_props(self, props):
        self.props.update(props)

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
        cls = type(self)
        newctx = cls.__new__(cls)
        newctx._import_fields(self)

        if bindings:
            newctx.bindings = self.bindings.copy()
            newctx.bindings.update(bindings)
        else:
            # share the dictionary
            newctx.bindings = self.bindings

        if props:
            newctx.props = self.props.copy()
            newctx._update_props(props)
        else:
            # share the dictionary
            newctx.props = self.props

        return newctx


IEEE_wp = {}
IEEE_wp.update((k, (5, 11)) for k in binary16_synonyms)
IEEE_wp.update((k, (8, 24)) for k in binary32_synonyms)
IEEE_wp.update((k, (11, 53)) for k in binary64_synonyms)
IEEE_wp.update((k, (15, 113)) for k in binary128_synonyms)

IEEE_rm = {}
IEEE_rm.update((k, RM.RNE) for k in RNE_synonyms)
IEEE_rm.update((k, RM.RNA) for k in RNA_synonyms)
IEEE_rm.update((k, RM.RTP) for k in RTP_synonyms)
IEEE_rm.update((k, RM.RTN) for k in RTN_synonyms)
IEEE_rm.update((k, RM.RTZ) for k in RTZ_synonyms)
IEEE_rm.update((k, RM.RAZ) for k in RAZ_synonyms)

class IEEECtx(EvalCtx):
    """Context for IEEE 754-like arithmetic."""

    w = 11
    p = 53
    rm = RM.RNE
    emax = (1 << (w - 1)) - 1
    emin = 1 - emax
    n = emin - p
    fbound = gmpmath.ieee_fbound(w, p)

    def __init__(self, w=None, p=None, rm=None, bindings=None, props=None):
        if bindings:
            self.bindings = bindings.copy()
        else:
            self.bindings = {}

        self.props = {}
        if props:
            self._update_props(props)

        # other keyword arguments can override the context
        if w is not None:
            self.w = w
        if p is not None:
            self.p = p
        if rm is not None:
            self.rm = rm
        self.emax = (1 << (self.w - 1)) - 1
        self.emin = 1 - self.emax
        self.n = self.emin - self.p
        self.fbound = gmpmath.ieee_fbound(self.w, self.p)

    def _update_props(self, props):
        if 'round' in props:
            try:
                self.rm = IEEE_rm[str(props['round']).strip().lower()]
            except KeyError:
                raise ValueError('unsupported rounding mode {}'.format(repr(props['round'])))
        if 'precision' in props:
            try:
                w, p = IEEE_wp[str(props['precision']).strip().lower()]
            except KeyError:
                precsym = str(props['precision']).strip().lower()
                if precsym.startswith('custom_binary'):
                    spec = precsym[13:]
                    specbits, specw = spec.split('_')
                    nbits = int(specbits)
                    w = int(specw)
                    p = nbits - w
                else:
                    precl = props['precision'].as_list()
                    if precl is not None and len(precl) == 3 and str(precl[0]) == 'float':
                        w = int(str(precl[1]))
                        p = int(str(precl[2]))
                    else:
                        raise ValueError('unsupported precision {}'.format(repr(props['precision'])))
            if w != self.w or p != self.p:
                self.w = w
                self.p = p
                self.emax = (1 << (w - 1)) - 1
                self.emin = 1 - self.emax
                self.n = self.emin - p
                self.fbound = gmpmath.ieee_fbound(w, p)
        self.props.update(props)

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


posit_esnbits = {}
posit_esnbits.update((k, (0, 8)) for k in binary8_synonyms)
posit_esnbits.update((k, (1, 16)) for k in binary16_synonyms)
posit_esnbits.update((k, (2, 32)) for k in binary32_synonyms)
posit_esnbits.update((k, (4, 64)) for k in binary64_synonyms)
posit_esnbits.update((k, (7, 128)) for k in binary128_synonyms)

# John Gustafson's Posits
class PositCtx(EvalCtx):
    """Context for John Gustafson's posit arithmetic."""

    es = 4
    nbits = 64
    u = 1 << es
    emax = u * (nbits - 2)
    emin = -emax
    p = nbits - 1

    def __init__(self, es=None, nbits=None, bindings=None, props=None):
        if bindings:
            self.bindings = bindings.copy()
        else:
            self.bindings = {}

        self.props = {}
        if props:
            self._update_props(props)

        # other keywords override the context
        if es is not None:
            self.es = es
        if nbits is not None:
            self.nbits = nbits
        self.u = 1 << self.es
        self.emax = self.u * (self.nbits - 2)
        self.emin = -self.emax
        self.p = self.nbits - 1

    def _update_props(self, props):
        if 'precision' in props:
            try:
                es, nbits = posit_esnbits[str(props['precision']).strip().lower()]
            except KeyError:
                precl = props['precision'].as_list()
                if precl is not None and len(precl) == 3 and str(precl[0]) == 'posit':
                    es = int(str(precl[1]))
                    nbits = int(str(precl[2]))
                else:
                    raise ValueError('unsupported precision {}'.format(repr(props['precision'])))
            if es != self.es or nbits != self.nbits:
                self.es = es
                self.nbits = nbits
                self.u = 1 << es
                self.emax = self.u * (self.nbits - 2)
                self.emin = -self.emax
                self.p = self.nbits - 1
        self.props.update(props)

    def _import_fields(self, ctx):
        self.es = ctx.es
        self.nbits = ctx.nbits
        self.u = ctx.u
        self.emax = ctx.emax
        self.emin = ctx.emin
        self.p = ctx.p

    def __repr__(self):
        args = ['es=' + repr(self.es), 'nbits=' + repr(self.nbits)]
        if len(self.bindings) > 0:
            args.append('bindings=' + repr(self.bindings))
        if len(self.props) > 0:
            args.append('props=' + repr(self.props))
        return '{}({})'.format(type(self).__name__, ', '.join(args))
