"""Emulated fixed-point arithmetic, also useful for quires.
"""

from ..titanic import digital
from ..titanic import gmpmath
from ..fpbench import fpcast as ast

from . import evalctx
from . import mpnum
from . import ieee754
from . import posit
from . import fixed
from . import interpreter

class MPMF(mpnum.MPNum):

    _ctx : evalctx.EvalCtx = ieee754.ieee_ctx(11, 53)

    @property
    def ctx(self):
        return self._ctx

    def as_ctx(self):
        ctx = self.ctx
        if isinstance(ctx, evalctx.IEEECtx):
            return ieee754.Float(self, ctx=ctx)
        elif isinstance(ctx, evalctx.PositCtx):
            return posit.Posit(self, ctx=ctx)
        elif isinstance(ctx, evalctx.FixedCtx):
            return fixed.Fixed(self, ctx=ctx)
        else:
            # TODO: ?
            # raise ValueError('unknown context {}'.format(repr(ctx)))
            return digital.Digital(self)

    def is_identical_to(self, other):
        return self.as_ctx().is_identical_to(other)

    def __init__(self, x=None, ctx=None, **kwargs):
        if ctx is None:
            ctx = type(self)._ctx

        if x is None or isinstance(x, digital.Digital):
            super().__init__(x=x, **kwargs)
        else:
            if kwargs:
                raise ValueError('cannot specify additional values {}'.format(repr(kwargs)))
            f = gmpmath.mpfr(x, ctx.p)
            unrounded = gmpmath.mpfr_to_digital(f)
            super().__init__(x=self._round_to_context(unrounded, ctx=ctx, strict=True))

        if isinstance(ctx, evalctx.IEEECtx):
            self._ctx = ieee754.ieee_ctx(ctx.es, ctx.nbits, rm=ctx.rm)
        elif isinstance(ctx, evalctx.PositCtx):
            self._ctx = posit.posit_ctx(ctx.es, ctx.nbits)
        elif isinstance(ctx, evalctx.FixedCtx):
            self._ctx = fixed.fixed_ctx(ctx.scale, ctx.nbits, rm=ctx.rm, of=ctx.of)
        else:
            raise ValueError('unsupported context {}'.format(repr(ctx)))

    def __repr__(self):
        return '{}(negative={}, c={}, exp={}, inexact={}, rc={}, isinf={}, isnan={}, ctx={})'.format(
            type(self).__name__, repr(self._negative), repr(self._c), repr(self._exp),
            repr(self._inexact), repr(self._rc), repr(self._isinf), repr(self._isnan), repr(self._ctx)
        )

    def __str__(self):
        return str(gmpmath.digital_to_mpfr(self))

    def __float__(self):
        return float(gmpmath.digital_to_mpfr(self))

    @classmethod
    def _select_context(cls, *args, ctx=None):
        if ctx is None:
            p = -1
            for f in args:
                if isinstance(f, cls) and f.ctx.p > p:
                    p = f.ctx.p
                    ctx = f.ctx

        if ctx is None:
            raise ValueError('arguments do not contain a context?\n{}'.format(repr(args)))

        if isinstance(ctx, evalctx.IEEECtx):
            return ieee754.ieee_ctx(ctx.es, ctx.nbits, rm=ctx.rm)
        elif isinstance(ctx, evalctx.PositCtx):
            return posit.posit_ctx(ctx.es, ctx.nbits)
        elif isinstance(ctx, evalctx.FixedCtx):
            return fixed.fixed_ctx(ctx.scale, ctx.nbits, rm=ctx.rm, of=ctx.of)
        else:
            raise ValueError('unsupported context {}'.format(repr(ctx)))

    @classmethod
    def _round_to_context(cls, unrounded, ctx=None, strict=False):
        if ctx is None:
            if hasattr(unrounded, 'ctx'):
                ctx = unrounded.ctx
            else:
                raise ValueError('unable to determine context to round {}'.format(repr(unrounded)))

        if isinstance(ctx, evalctx.IEEECtx):
            rounded = ieee754.Float._round_to_context(unrounded, ctx=ctx, strict=strict)
        elif isinstance(ctx, evalctx.PositCtx):
            rounded = posit.Posit._round_to_context(unrounded, ctx=ctx, strict=strict)
        elif isinstance(ctx, evalctx.FixedCtx):
            rounded = fixed.Fixed._round_to_context(unrounded, ctx=ctx, strict=strict)
        else:
            raise ValueError('unsupported context {}'.format(repr(ctx)))

        return cls(rounded, ctx=ctx)

    def isnormal(self):
        x = self.as_ctx()
        if isinstance(x, mpnum.MPNum):
            return x.isnormal()
        else:
            return  not (
                self.is_zero()
                or self.isinf
                or self.isnan
            )

# TODO: hack, provide a fake constructor-like thing to make contexts of varying types

def mpmf_ctype(bindings=None, props=None):
    ctx = MPMF._ctx.let(bindings=bindings)
    return evalctx.determine_ctx(ctx, props)

class Interpreter(interpreter.StandardInterpreter):
    dtype = MPMF
    ctype = mpmf_ctype

    @classmethod
    def arg_to_digital(cls, x, ctx):
        return cls.dtype(x, ctx=ctx)

    @classmethod
    def _eval_constant(cls, e, ctx):
        return cls.round_to_context(gmpmath.compute_constant(e.value, prec=ctx.p), ctx=ctx)

    # unfortunately, interpreting these values efficiently requries info from the context,
    # so it has to be implemented per interpreter...

    @classmethod
    def _eval_integer(cls, e, ctx):
        x = digital.Digital(m=e.i, exp=0, inexact=False)
        return cls.round_to_context(x, ctx=ctx)

    @classmethod
    def _eval_rational(cls, e, ctx):
        p = digital.Digital(m=e.p, exp=0, inexact=False)
        q = digital.Digital(m=e.q, exp=0, inexact=False)
        x = gmpmath.compute(OP.div, p, q, prec=ctx.p)
        return cls.round_to_context(x, ctx=ctx)

    @classmethod
    def _eval_digits(cls, e, ctx):
        x = gmpmath.compute_digits(e.m, e.e, e.b, prec=ctx.p)
        return cls.round_to_context(x, ctx=ctx)

    # this is what makes it mpmf actually
    @classmethod
    def _eval_ctx(cls, e, ctx):
        return cls.evaluate(e.body, evalctx.determine_ctx(ctx, e.props))

    @classmethod
    def round_to_context(cls, x, ctx):
        """Not actually used???"""
        return cls.dtype._round_to_context(x, ctx=ctx, strict=False)

    # copy-pasta hack
    @classmethod
    def arg_ctx(cls, core, args, ctx=None, override=True):
        if len(core.inputs) != len(args):
            raise ValueError('incorrect number of arguments: got {}, expecting {} ({})'.format(
                len(args), len(core.inputs), ' '.join((name for name, props, shape in core.inputs))))

        if ctx is None:
            ctx = cls.ctype(props=core.props)
        elif override:
            allprops = {}
            allprops.update(core.props)
            allprops.update(ctx.props)
            ctx = evalctx.determine_ctx(ctx, allprops)
        else:
            ctx = evalctx.determine_ctx(ctx, core.props)

        arg_bindings = []

        for arg, (name, props, shape) in zip(args, core.inputs):
            local_ctx = evalctx.determine_ctx(ctx, props)

            if isinstance(arg, cls.dtype):
                argval = cls.round_to_context(arg, ctx=local_ctx)
            elif isinstance(arg, ast.Expr):
                argval = cls.evaluate(arg, local_ctx)
            else:
                argval = cls.arg_to_digital(arg, local_ctx)

            arg_bindings.append((name, argval))

        return ctx.let(bindings=arg_bindings)
