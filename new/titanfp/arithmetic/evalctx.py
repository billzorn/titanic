"""Evaluation context information, shared across arithmetics."""

import re

from ..titanic import utils
from ..titanic import gmpmath
from ..titanic.ops import RM, OF


binary8_synonyms = {'binary8', 'posit8', 'posit8_t', 'quad'}
binary16_synonyms = {'binary16', 'float16', 'float16_t', 'posit16', 'posit16_t', 'half'}
binary32_synonyms = {'binary32', 'float32', 'float32_t', 'posit32', 'posit32_t', 'single', 'float'}
binary64_synonyms = {'binary64', 'float64', 'float64_t', 'posit64', 'posit64_t', 'double'}
binary80_synonyms = {'binary80', 'float80', 'extended', 'longdouble'}
binary128_synonyms = {'binary128', 'float128', 'float128_t', 'posit128', 'posit128_t', 'quadruple'}

int8_synonyms = {'int8', 'int8_t', 'char'}
int16_synonyms = {'int16', 'int16_t', 'short'}
int32_synonyms = {'int32', 'int32_t', 'int'}
int64_synonyms = {'int64', 'int64_t', 'long'}

RNE_synonyms = {'rne', 'nearesteven', 'roundnearesteven', 'nearesttiestoeven', 'roundnearesttiestoeven'}
RNA_synonyms = {'rna', 'nearestaway', 'roundnearestaway', 'nearesttiestoaway', 'roundnearesttiestoaway'}
RTP_synonyms = {'rtp', 'topositive', 'roundtopositive', 'towardpositive', 'roundtowardpositive'}
RTN_synonyms = {'rtn', 'tonegative', 'roundtonegative', 'towardnegative', 'roundtowardnegative'}
RTZ_synonyms = {'rtz', 'tozero', 'roundtozero', 'towardzero', 'roundtowardzero'}
RAZ_synonyms = {'raz', 'awayzero', 'roundawayzero', 'awayzero', 'roundawayzero'}

infinity_synonyms = {'infinity', 'inf'}
clamp_synonyms = {'clamp'}
wrap_synonyms = {'wrap'}

prefer_float = {'binary', 'float', 'quad', 'half', 'single', 'double', 'extended', 'longdouble', 'quadruple'}
prefer_posit = {'posit'}
prefer_fixed = {'int', 'char', 'short', 'long'}


class EvalCtx(object):
    """Generic context for holding variable bindings and properties."""

    # these placeholders should never have anything put in them
    bindings = utils.ImmutableDict()
    props = utils.ImmutableDict()

    def __init__(self, bindings=None, props=None):
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

    _reserved_fields = {'bindings', 'props'}

    def __str__(self):
        fields = ['    ' + str(k) + ': ' + str(v) for k, v in self.__dict__.items() if k not in _reserved_fields]
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


IEEE_esnbits = {}
IEEE_esnbits.update((k, (5, 16)) for k in binary16_synonyms)
IEEE_esnbits.update((k, (8, 32)) for k in binary32_synonyms)
IEEE_esnbits.update((k, (11, 64)) for k in binary64_synonyms)
IEEE_esnbits.update((k, (15, 79)) for k in binary80_synonyms)
IEEE_esnbits.update((k, (15, 128)) for k in binary128_synonyms)

IEEE_rm = {}
IEEE_rm.update((k, RM.RNE) for k in RNE_synonyms)
IEEE_rm.update((k, RM.RNA) for k in RNA_synonyms)
IEEE_rm.update((k, RM.RTP) for k in RTP_synonyms)
IEEE_rm.update((k, RM.RTN) for k in RTN_synonyms)
IEEE_rm.update((k, RM.RTZ) for k in RTZ_synonyms)
IEEE_rm.update((k, RM.RAZ) for k in RAZ_synonyms)

class IEEECtx(EvalCtx):
    """Context for IEEE 754-like arithmetic."""

    es = 11
    nbits = 64
    rm = RM.RNE

    p = nbits - es
    emax = (1 << (es - 1)) - 1
    emin = 1 - emax
    n = emin - p
    fbound = gmpmath.ieee_fbound(es, p)

    def __init__(self, bindings=None, props=None, es=None, nbits=None, rm=None):
        init_es = self.es
        init_nbits = self.nbits
        init_rm = self.rm

        if bindings:
            self.bindings = bindings.copy()
        else:
            self.bindings = {}

        self.props = {}
        if props:
            # reproduce (slightly modified) code for efficiency
            if 'round' in props:
                rounding = props['round']
                try:
                    init_rm = IEEE_rm[str(rounding).lower()]
                except KeyError:
                        # these should all be custom exceptions
                        raise ValueError('unsupported IEEE 754 rounding mode {}'.format(repr(rounding)))

            if 'precision' in props:
                prec = props['precision']
                precstr = str(prec).lower()
                if precstr in IEEE_esnbits:
                    init_es, init_nbits = IEEE_esnbits[precstr]
                else:
                    # try to decipher custom type
                    try:
                        precl = prec.as_list()
                        assert str(precl[0]).lower() == 'float'
                        init_es = int(str(precl[1]))
                        init_nbits = int(str(precl[2]))
                    except Exception:
                        raise ValueError('unsupported IEEE 754 precision {}'.format(repr(prec)))

            self.props.update(props)

        # arguments are allowed to override properties
        if es is not None:
            init_es = es
        if nbits is not None:
            init_nbits = nbits
        if rm is not None:
            init_rm = rm

        self.rm = init_rm

        if init_es != self.es or init_nbits != self.nbits:
            self.es = init_es
            self.nbits = init_nbits
            self.p = self.nbits - self.es
            self.emax = (1 << (self.es - 1)) - 1
            self.emin = 1 - self.emax
            self.n = self.emin - self.p
            self.fbound = gmpmath.ieee_fbound(self.es, self.p)
        else:
            self.es = self.es
            self.nbits = self.nbits
            self.p = self.p
            self.emax = self.emax
            self.emin = self.emin
            self.n = self.n
            self.fbound = self.fbound

    def _update_props(self, props):
        init_es = self.es
        init_nbits = self.nbits
        init_rm = self.rm

        if 'round' in props:
            rounding = props['round']
            try:
                init_rm = IEEE_rm[str(rounding).lower()]
            except KeyError:
                    # these should all be custom exceptions
                    raise ValueError('unsupported IEEE 754 rounding mode {}'.format(repr(rounding)))

        if 'precision' in props:
            prec = props['precision']
            precstr = str(prec).lower()
            if precstr in IEEE_esnbits:
                init_es, init_nbits = IEEE_esnbits[precstr]
            else:
                # try to decipher custom type
                try:
                    precl = prec.as_list()
                    assert str(precl[0]).lower() == 'float'
                    init_es = int(str(precl[1]))
                    init_nbits = int(str(precl[2]))
                except Exception:
                    raise ValueError('unsupported IEEE 754 precision {}'.format(repr(prec)))

        self.props.update(props)

        self.rm = init_rm

        if init_es != self.es or init_nbits != self.nbits:
            self.es = init_es
            self.nbits = init_nbits
            self.p = self.nbits - self.es
            self.emax = (1 << (self.es - 1)) - 1
            self.emin = 1 - self.emax
            self.n = self.emin - self.p
            self.fbound = gmpmath.ieee_fbound(self.es, self.p)


    def _import_fields(self, ctx):
        self.rm = ctx.rm
        self.es = ctx.es
        self.nbits = ctx.nbits
        self.p = ctx.p
        self.emax = ctx.emax
        self.emin = ctx.emin
        self.n = ctx.n
        self.fbound = ctx.fbound

    def __repr__(self):
        args = []
        if len(self.bindings) > 0:
            args.append('bindings=' + repr(self.bindings))
        if len(self.props) > 0:
            args.append('props=' + repr(self.props))
        args += ['es=' + repr(self.es), 'nbits=' + repr(self.nbits), 'rm=' + str(self.rm)]
        return '{}({})'.format(type(self).__name__, ', '.join(args))


posit_esnbits = {}
posit_esnbits.update((k, (0, 8)) for k in binary8_synonyms)
posit_esnbits.update((k, (1, 16)) for k in binary16_synonyms)
posit_esnbits.update((k, (2, 32)) for k in binary32_synonyms)
posit_esnbits.update((k, (3, 64)) for k in binary64_synonyms)
posit_esnbits.update((k, (4, 80)) for k in binary80_synonyms)
posit_esnbits.update((k, (7, 128)) for k in binary128_synonyms)

# John Gustafson's Posits
class PositCtx(EvalCtx):
    """Context for John Gustafson's posit arithmetic."""

    es = 3
    nbits = 64

    p = nbits - 1 - es
    u = 1 << es
    emax = u * (nbits - 2)
    emin = -emax

    def __init__(self, bindings=None, props=None, es=None, nbits=None):
        init_es = self.es
        init_nbits = self.nbits

        if bindings:
            self.bindings = bindings.copy()
        else:
            self.bindings = {}

        self.props = {}
        if props:
            # reproduce (slightly modified) code for efficiency
            if 'precision' in props:
                prec = props['precision']
                precstr = str(prec).lower()
                if precstr in posit_esnbits:
                    init_es, init_nbits = posit_esnbits[precstr]
                else:
                    # try to decipher custom type
                    try:
                        precl = prec.as_list()
                        assert str(precl[0]).lower() == 'posit'
                        init_es = int(str(precl[1]))
                        init_nbits = int(str(precl[2]))
                    except Exception:
                        raise ValueError('unsupported posit precision {}'.format(repr(prec)))

            self.props.update(props)

        # arguments are allowed to override properties
        if es is not None:
            init_es = es
        if nbits is not None:
            init_nbits = nbits

        if init_es != self.es or init_nbits != self.nbits:
            self.es = init_es
            self.nbits = init_nbits
            self.p = self.nbits - 1 - self.es
            self.u = 1 << self.es
            self.emax = self.u * (self.nbits - 2)
            self.emin = -self.emax
        else:
            self.es = self.es
            self.nbits = self.nbits
            self.p = self.p
            self.u = self.u
            self.emax = self.emax
            self.emin = self.emin

    def _update_props(self, props):
        init_es = self.es
        init_nbits = self.nbits

        if 'precision' in props:
            prec = props['precision']
            precstr = str(prec).lower()
            if precstr in posit_esnbits:
                init_es, init_nbits = posit_esnbits[precstr]
            else:
                # try to decipher custom type
                try:
                    precl = prec.as_list()
                    assert str(precl[0]).lower() == 'posit'
                    init_es = int(str(precl[1]))
                    init_nbits = int(str(precl[2]))
                except Exception:
                    raise ValueError('unsupported posit precision {}'.format(repr(prec)))

        self.props.update(props)

        if init_es != self.es or init_nbits != self.nbits:
            self.es = init_es
            self.nbits = init_nbits
            self.p = self.nbits - 1 - self.es
            self.u = 1 << self.es
            self.emax = self.u * (self.nbits - 2)
            self.emin = -self.emax

    def _import_fields(self, ctx):
        self.es = ctx.es
        self.nbits = ctx.nbits
        self.p = ctx.p
        self.u = ctx.u
        self.emax = ctx.emax
        self.emin = ctx.emin

    def __repr__(self):
        args = []
        if len(self.bindings) > 0:
            args.append('bindings=' + repr(self.bindings))
        if len(self.props) > 0:
            args.append('props=' + repr(self.props))
        args += ['es=' + repr(self.es), 'nbits=' + repr(self.nbits)]
        return '{}({})'.format(type(self).__name__, ', '.join(args))


fixed_snbits = {}
fixed_snbits.update((k, (0, 8)) for k in int8_synonyms)
fixed_snbits.update((k, (0, 16)) for k in int16_synonyms)
fixed_snbits.update((k, (0, 32)) for k in int32_synonyms)
fixed_snbits.update((k, (0, 64)) for k in int64_synonyms)

fixed_of = {}
fixed_of.update((k, OF.INFINITY) for k in infinity_synonyms)
fixed_of.update((k, OF.WRAP) for k in wrap_synonyms)
fixed_of.update((k, OF.CLAMP) for k in clamp_synonyms)

class FixedCtx(EvalCtx):
    """Context for fixed point arithmetic, including bounded integers and quires."""

    scale = 0
    nbits = 64
    rm = RM.RTN
    of = OF.WRAP

    p = nbits
    n = scale - 1

    def __init__(self, bindings=None, props=None, scale=None, nbits=None, rm=None, of=None):
        init_scale = self.scale
        init_nbits = self.nbits
        init_rm = self.rm
        init_of = self.of

        if bindings:
            self.bindings = bindings.copy()
        else:
            self.bindings = {}

        self.props = {}
        if props:
            if 'round' in props:
                rounding = props['round']
                try:
                    init_rm = IEEE_rm[str(rounding).lower()]
                except KeyError:
                        # these should all be custom exceptions
                        raise ValueError('unsupported fixed-point rounding mode {}'.format(repr(rounding)))

            if 'overflow' in props:
                overflow = props['overflow']
                try:
                    init_of = fixed_of[str(overflow).lower()]
                except KeyError:
                        # these should all be custom exceptions
                        raise ValueError('unsupported fixed-point overflow mode {}'.format(repr(overflow)))

            if 'precision' in props:
                prec = props['precision']
                precstr = str(prec).lower()
                if precstr in fixed_snbits:
                    init_scale, init_nbits = fixed_snbits[precstr]
                else:
                    # try to decipher custom type
                    try:
                        precl = prec.as_list()
                        assert str(precl[0]).lower() == 'fixed'
                        init_scale = int(str(precl[1]))
                        init_nbits = int(str(precl[2]))
                    except Exception:
                        raise ValueError('unsupported fixed-point precision {}'.format(repr(prec)))

            self.props.update(props)

        if scale is not None:
            init_scale = scale
        if nbits is not None:
            init_nbits = nbits
        if rm is not None:
            init_rm = rm
        if of is not None:
            init_of = of

        # don't even need to test here, there is no expensive recomputation
        self.scale = init_scale
        self.nbits = init_nbits
        self.rm = init_rm
        self.of = init_of
        self.p = self.nbits
        self.n = self.scale - 1

    def _update_props(self, props):
        init_scale = self.scale
        init_nbits = self.nbits
        init_rm = self.rm
        init_of = self.of

        if 'round' in props:
            rounding = props['round']
            try:
                init_rm = IEEE_rm[str(rounding).lower()]
            except KeyError:
                    # these should all be custom exceptions
                    raise ValueError('unsupported fixed-point rounding mode {}'.format(repr(rounding)))

        if 'overflow' in props:
            overflow = props['overflow']
            try:
                init_of = fixed_of[str(overflow).lower()]
            except KeyError:
                    # these should all be custom exceptions
                    raise ValueError('unsupported fixed-point overflow mode {}'.format(repr(overflow)))

        if 'precision' in props:
            prec = props['precision']
            precstr = str(prec).lower()
            if precstr in fixed_snbits:
                init_scale, init_nbits = fixed_snbits[precstr]
            else:
                # try to decipher custom type
                try:
                    precl = prec.as_list()
                    assert str(precl[0]).lower() == 'fixed'
                    init_scale = int(str(precl[1]))
                    init_nbits = int(str(precl[2]))
                except Exception:
                    raise ValueError('unsupported fixed-point precision {}'.format(repr(prec)))

        self.props.update(props)

        self.scale = init_scale
        self.nbits = init_nbits
        self.rm = init_rm
        self.of = init_of
        self.p = self.nbits
        self.n = self.scale - 1

    def _import_fields(self, ctx):
        self.scale = ctx.scale
        self.nbits = ctx.nbits
        self.rm = ctx.rm
        self.of = ctx.of
        self.p = ctx.p
        self.n = ctx.n

    def __repr__(self):
        args = []
        if len(self.bindings) > 0:
            args.append('bindings=' + repr(self.bindings))
        if len(self.props) > 0:
            args.append('props=' + repr(self.props))
        args += ['scale=' + repr(self.scale), 'nbits=' + repr(self.nbits), 'rm=' + repr(self.rm), 'of=' + repr(self.of)]
        return '{}({})'.format(type(self).__name__, ', '.join(args))


ctx_type_re = re.compile(r'^\s*[(]?\s*(?:(' +
                         r'|'.join(prefer_float) + r')|(' +
                         r'|'.join(prefer_posit) + ')|(' +
                         r'|'.join(prefer_fixed) + r'))')

def determine_ctx(old_ctx, props):
    if 'precision' in props:
        prec = props['precision']
        m = ctx_type_re.match(str(prec))
        print(m)
        if m:
            if m.group(1):
                new_ctx_t = IEEECtx
            elif m.group(2):
                new_ctx_t = PositCtx
            else:
                new_ctx_t = FixedCtx
        else:
            raise ValueError('unsupported precision annotation {}'.format(prec))

    else:
        new_ctx_t = type(old_ctx)

    # TODO: implement automatic quire sizing here

    if isinstance(old_ctx, new_ctx_t):
        return old_ctx.let(props=props)
    else:
        # # this would explode because the olds props is problematic for the new constructor
        # new_ctx = new_ctx_t(bindings=old_ctx.bindings, props=old_ctx.props)
        # return new_ctx.let(props=props)

        new_props = old_ctx.props.copy()
        new_props.update(props)
        return new_ctx_t(bindings=old_ctx.bindings, props=new_props)
