"""Emulated IEEE 754 floating-point arithmetic.
"""

from ..titanic import gmpmath
from ..titanic import digital

from ..titanic.integral import bitmask
from ..titanic.ops import RM, OP
from .evalctx import IEEECtx
from . import interpreter


used_ctxs = {}
def ieee_ctx(w, p):
    try:
        return used_ctxs[(w, p)]
    except KeyError:
        ctx = IEEECtx(w=w, p=p)
        used_ctxs[(w, p)] = ctx
        return ctx


class Float(digital.Digital):

    _ctx : IEEECtx = ieee_ctx(11, 53)

    @property
    def ctx(self):
        """The rounding context used to compute this value.
        If a computation takes place between two values, then
        it will either use a provided context (which will be recorded
        on the result) or the more precise of the parent contexts
        if none is provided.
        """
        return self._ctx

    def is_identical_to(self, other):
        if isinstance(other, Float):
            return super().is_identical_to(other) and self.ctx.w == other.ctx.w and self.ctx.p == other.ctx.p
        else:
            return super().is_identical_to(other)

    def __init__(self, x=None, ctx=None, **kwargs):
        if ctx is None:
            ctx = type(self)._ctx

        if x is None or isinstance(x, digital.Digital):
            super().__init__(x=x, **kwargs)
        else:
            f = gmpmath.mpfr(x, ctx.p)
            unrounded = gmpmath.mpfr_to_digital(f)
            super().__init__(x=self._round_to_context(unrounded, ctx=ctx, strict=True), **kwargs)

        self._ctx = ctx

    def __str__(self):
        return str(gmpmath.digital_to_mpfr(self))

    @staticmethod
    def _select_context(*ctxs, ctx=None):
        if ctx is not None:
            return ieee_ctx(ctx.w, ctx.p)
        else:
            w = max((c.w for c in ctxs))
            p = max((c.p for c in ctxs))
            return ieee_ctx(w, p)

    @classmethod
    def _round_to_context(cls, unrounded, ctx=None, strict=False):
        if ctx is None:
            if hasattr(unrounded, 'ctx'):
                ctx = unrounded.ctx
            else:
                raise ValueError('no context specified to round {}'.format(repr(unrounded)))

        if ctx.rm != RM.RNE:
            raise ValueError('unimplemented rounding mode {}'.format(repr(rm)))

        if unrounded.isinf or unrounded.isnan:
            return cls(unrounded, ctx=ctx)

        magnitude = cls(unrounded, negative=False)
        if magnitude > ctx.fbound:
            return cls(negative=unrounded.negative, isinf=True, ctx=ctx)
        else:
            return cls(unrounded.round_m(max_p=ctx.p, min_n=ctx.n, rm=ctx.rm, strict=strict), ctx=ctx)
    
    # operations
    
    def add(self, other, ctx=None):
        ctx = self._select_context(self.ctx, other.ctx, ctx=ctx)
        result = gmpmath.compute(OP.add, self, other, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def sub(self, other, ctx=None):
        ctx = self._select_context(self.ctx, other.ctx, ctx=ctx)
        result = gmpmath.compute(OP.sub, self, other, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def mul(self, other, ctx=None):
        ctx = self._select_context(self.ctx, other.ctx, ctx=ctx)
        result = gmpmath.compute(OP.mul, self, other, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def div(self, other, ctx=None):
        ctx = self._select_context(self.ctx, other.ctx, ctx=ctx)
        result = gmpmath.compute(OP.div, self, other, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def sqrt(self, ctx=None):
        ctx = self._select_context(self.ctx, ctx=ctx)
        result = gmpmath.compute(OP.sqrt, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)

    def fma(self, other1, other2, ctx=None):
        ctx = self._select_context(self.ctx, other1.ctx, other2.ctx, ctx=ctx)
        result = gmpmath.compute(OP.fma, self, other1, other2, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)
    
    def neg(self, ctx=None):
        ctx = self._select_context(self.ctx, ctx=ctx)
        result = gmpmath.compute(OP.neg, self, prec=ctx.p)
        return self._round_to_context(result, ctx=ctx, strict=True)


class Interpreter(interpreter.StandardInterpreter):
    dtype = Float
    ctype = IEEECtx

    @classmethod
    def arg_to_digital(cls, x, ctx):
        return cls.dtype(x, ctx=ctx)

    @classmethod
    def round_to_context(cls, x, ctx):
        """Not actually used?"""
        return cls.dtype._round_to_context(x, ctx=ctx, strict=False)


USE_GMP = True
USE_MATH = False
DEFAULT_IEEE_CTX = IEEECtx(w=11, p=53, rm=RM.RNE) # double w/ RNE

def compute_with_backend(opcode, *args, prec=54):
    result = None
    if USE_MATH:
        result_math = wolfmath.compute(opcode, *args, prec=prec)
        result = result_math
    if USE_GMP:
        result_gmp = gmpmath.compute(opcode, *args, prec=prec)
        result = result_gmp
    if USE_GMP and USE_MATH:
        if not result_gmp.is_identical_to(result_math):
            print(repr(result_gmp))
            print(repr(result_math))
            print('--')
    if result is None:
        raise ValueError('no backend specified')
    return result


def round_to_ieee_ctx(x, ctx=DEFAULT_IEEE_CTX):
    if x.isinf or x.isnan:
        # no rounding to perform
        return sinking.Sink(x)

    x_emag = sinking.Sink(x, negative=False, inexact=False)

    if ctx.rm == RM.RNE:
        if x_emag >= ctx.fbound:
            return sinking.Sink(negative=x.negative, c=0, exp=0, inf=True, rc=-1)
    else:
        raise ValueError('round_to_ieee_ctx: unsupported rounding mode {}'.format(repr(ctx.rm)))

    return x.round_m(max_p=ctx.p, min_n=ctx.n)


def arg_to_digital(x, ctx=DEFAULT_IEEE_CTX):
    result = gmpmath.mpfr_to_digital(gmpmath.mpfr(x, ctx.p + 1))
    return round_to_ieee_ctx(result, ctx=ctx)


def digital_to_bits(x, ctx=DEFAULT_IEEE_CTX):
    if ctx.p < 2 or ctx.w < 2:
        raise ValueError('format with w={}, p={} cannot be represented with IEEE 754 bit pattern'.format(ctx.w, ctx.p))

    try:
        rounded = round_to_ieee_ctx(x, ctx)
    except sinking.PrecisionError:
        rounded = round_to_ieee_ctx(sinking.Sink(x, inexact=False), ctx)

    pbits = ctx.p - 1

    if rounded.negative:
        S = 1
    else:
        S = 0

    if rounded.isnan:
        # canonical NaN
        return (0 << (ctx.w + pbits)) | (bitmask(ctx.w) << pbits) | (1 << (pbits - 1))
    elif rounded.isinf:
        return (S << (ctx.w + pbits)) | (bitmask(ctx.w) << pbits) # | 0
    elif rounded.is_zero():
        return (S << (ctx.w + pbits)) # | (0 << pbits) | 0

    c = rounded.c
    cbits = rounded.p
    e = rounded.e

    if e < ctx.emin:
        # subnormal
        lz = (ctx.emin - 1) - e
        if lz > pbits or (lz == pbits and cbits > 0):
            raise ValueError('exponent out of range: {}'.format(e))
        elif lz + cbits > pbits:
            raise ValueError('too much precision: given {}, can represent {}'.format(cbits, pbits - lz))
        E = 0
        C = c << (lz - (pbits - cbits))
    elif e <= ctx.emax:
        # normal
        if cbits > ctx.p:
            raise ValueError('too much precision: given {}, can represent {}'.format(cbits, ctx.p))
        elif cbits < ctx.p:
            raise ValueError('too little precision: given {}, can represent {}'.format(cbits, ctx.p))
        E = e + ctx.emax
        C = (c << (ctx.p - cbits)) & bitmask(pbits)
    else:
        # overflow
        raise ValueError('exponent out of range: {}'.format(e))

    return (S << (ctx.w + pbits)) | (E << pbits) | C


def bits_to_digital(i, ctx=DEFAULT_IEEE_CTX):
    pbits = ctx.p - 1

    S = (i >> (ctx.w + pbits)) & bitmask(1)
    E = (i >> pbits) & bitmask(ctx.w)
    C = i & bitmask(pbits)

    negative = (S == 1)
    e = E - ctx.emax

    if E == 0:
        # subnormal
        c = C
        exp = -ctx.emax - pbits + 1
    elif e <= ctx.emax:
        # normal
        c = C | (1 << pbits)
        exp = e - pbits
    else:
        # nonreal
        if C == 0:
            return sinking.Sink(negative=negative, c=0, exp=0, inf=True, rc=0)
        else:
            return sinking.Sink(negative=False, c=0, exp=0, nan=True, rc=0)

    # unfortunately any rc / exactness information is lost
    return sinking.Sink(negative=negative, c=c, exp=exp, inexact=False, rc=0)


def show_bitpattern(x, ctx=DEFAULT_IEEE_CTX):
    print(x)

    if isinstance(x, int):
        i = x
    elif isinstance(x, sinking.Sink):
        i = digital_to_bits(x, ctx=ctx)

    S = i >> (ctx.w + ctx.p - 1)
    E = (i >> (ctx.p - 1)) & bitmask(ctx.w)
    C = i & bitmask(ctx.p - 1)
    if E == 0 or E == bitmask(ctx.w):
        hidden = 0
    else:
        hidden = 1

    return ('float{:d}({:d},{:d}): {:01b} {:0'+str(ctx.w)+'b} ({:01b}) {:0'+str(ctx.p-1)+'b}').format(
        ctx.w + ctx.p, ctx.w, ctx.p, S, E, hidden, C,
    )


import numpy as np
import sys
def bits_to_numpy(i, nbytes=8, dtype=np.float64):
    return np.frombuffer(
        i.to_bytes(nbytes, sys.byteorder),
        dtype=dtype, count=1, offset=0,
    )[0]


def add(x1, x2, ctx):
    prec = max(2, ctx.p + 1)
    result = compute_with_backend(OP.add, x1, x2, prec=prec)
    return round_to_ieee_ctx(result, ctx)

def sub(x1, x2, ctx):
    prec = max(2, ctx.p + 1)
    result = compute_with_backend(OP.sub, x1, x2, prec=prec)
    return round_to_ieee_ctx(result, ctx)

def mul(x1, x2, ctx):
    prec = max(2, ctx.p + 1)
    result = compute_with_backend(OP.mul, x1, x2, prec=prec)
    return round_to_ieee_ctx(result, ctx)

def div(x1, x2, ctx):
    prec = max(2, ctx.p + 1)
    result = compute_with_backend(OP.div, x1, x2, prec=prec)
    return round_to_ieee_ctx(result, ctx)

def neg(x, ctx):
    prec = max(2, ctx.p + 1)
    result = compute_with_backend(OP.neg, x, prec=prec)
    return round_to_ieee_ctx(result, ctx)

def sqrt(x, ctx):
    prec = max(2, ctx.p + 1)
    result = compute_with_backend(OP.sqrt, x, prec=prec)
    return round_to_ieee_ctx(result, ctx)

def floor(x, ctx):
    prec = max(2, ctx.p + 1)
    result = compute_with_backend(OP.floor, x, prec=prec)
    return round_to_ieee_ctx(result, ctx)

def fmod(x1, x2, ctx):
    prec = max(2, ctx.p + 1)
    result = compute_with_backend(OP.fmod, x1, x2, prec=prec)
    return round_to_ieee_ctx(result, ctx)

def pow(x1, x2, ctx):
    prec = max(2, ctx.p + 1)
    result = compute_with_backend(OP.pow, x1, x2, prec=prec)
    return round_to_ieee_ctx(result, ctx)

def sin(x, ctx):
    prec = max(2, ctx.p + 1)
    result = compute_with_backend(OP.sin, x, prec=prec)
    return round_to_ieee_ctx(result, ctx)

def acos(x, ctx):
    prec = max(2, ctx.p + 1)
    result = compute_with_backend(OP.acos, x, prec=prec)
    return round_to_ieee_ctx(result, ctx)


def interpret(core, args, ctx=None):
    """FPCore interpreter for IEEE 754-like arithmetic."""

    if len(core.inputs) != len(args):
        raise ValueError('incorrect number of arguments: got {}, expecting {} ({})'
                         .format(len(args), len(core.inputs), ' '.join((name for name, props in core.inputs))))

    if ctx is None:
        ctx = IEEECtx(props=core.props)

    for arg, (name, props) in zip(args, core.inputs):
        if props:
            local_ctx = IEEECtx(props=props)
        else:
            local_ctx = ctx

        if isinstance(arg, sinking.Sink):
            argval = arg
        else:
            argval = arg_to_digital(arg, local_ctx)
        ctx.let([(name, argval)])

    return evaluate(core.e, ctx)


def interpret_pre(core, args, ctx=None):
    if core.pre is None:
        raise ValueError('core has no preconditions')

    if len(core.inputs) != len(args):
        raise ValueError('incorrect number of arguments: got {}, expecting {} ({})'
                         .format(len(args), len(core.inputs), ' '.join((name for name, props in core.inputs))))

    if ctx is None:
        ctx = IEEECtx(props=core.props)

    for arg, (name, props) in zip(args, core.inputs):
        if props:
            local_ctx = IEEECtx(props=props)
        else:
            local_ctx = ctx

        if isinstance(arg, sinking.Sink):
            argval = arg
        else:
            argval = arg_to_digital(arg, local_ctx)
        ctx.let([(name, argval)])

    return evaluate(core.pre, ctx)


def evaluate(e, ctx):
    """Recursive expression evaluator, with much isinstance()."""

    # ValueExpr

    if isinstance(e, ast.Val):
        s = e.value.upper()
        if s == 'TRUE':
            return True
        elif s == 'FALSE':
            return False
        else:
            return arg_to_digital(e.value, ctx)

    elif isinstance(e, ast.Var):
        # never rounded
        return ctx.bindings[e.value]

    # and Digits

    elif isinstance(e, ast.Digits):
        raise ValueError('unimplemented')
        # TODO yolo
        spare_bits = 16
        base = sinking.Sink(e.b)
        exponent = sinking.Sink(e.e)
        scale = gmpmath.pow(base, exponent,
                            min_n = -(base.bit_length() * exponent) - spare_bits,
                            max_p = ctx.w + ctx.p + spare_bits)
        significand = sinking.Sink(e.m,
                                   min_n = ctx.n - spare_bits,
                                   max_p = ctx.p + spare_bits)
        return gmpmath.mul(significand, scale, min_n=ctx.n, max_p=ctx.p)

    # control flow

    elif isinstance(e, ast.Ctx):
        newctx = IEEECtx(props=e.props)
        newctx.let(ctx.bindings)
        return evaluate(e.body, newctx)

    elif isinstance(e, ast.If):
        if evaluate(e.cond, ctx):
            return evaluate(e.then_body, ctx)
        else:
            return evaluate(e.else_body, ctx)

    elif isinstance(e, ast.Let):
        bindings = [(name, evaluate(expr, ctx)) for name, expr in e.let_bindings]
        newctx = ctx.clone().let(bindings)
        return evaluate(e.body, newctx)

    elif isinstance(e, ast.While):
        bindings = [(name, evaluate(init_expr, ctx)) for name, init_expr, update_expr in e.while_bindings]
        newctx = ctx.clone().let(bindings)
        while evaluate(e.cond, newctx):
            bindings = [(name, evaluate(update_expr, newctx)) for name, init_expr, update_expr in e.while_bindings]
            newctx = newctx.clone().let(bindings)
        return evaluate(e.body, newctx)

    # Unary/Binary/NaryExpr

    else:
        children = [evaluate(child, ctx) for child in e.children]

        if isinstance(e, ast.Neg):
            return neg(*children, ctx)

        elif isinstance(e, ast.Sqrt):
            return sqrt(*children, ctx)

        elif isinstance(e, ast.Add):
            return add(*children, ctx)

        elif isinstance(e, ast.Sub):
            return sub(*children, ctx)

        elif isinstance(e, ast.Mul):
            return mul(*children, ctx)

        elif isinstance(e, ast.Div):
            return div(*children, ctx)

        elif isinstance(e, ast.Floor):
            return floor(*children, ctx)

        elif isinstance(e, ast.Fmod):
            return fmod(*children, ctx)

        elif isinstance(e, ast.Pow):
            return pow(*children, ctx)

        elif isinstance(e, ast.Sin):
            return sin(*children, ctx)

        elif isinstance(e, ast.Acos):
            return acos(*children, ctx)

        elif isinstance(e, ast.LT):
            for x, y in zip(children, children[1:]):
                if not x < y:
                    return False
            return True

        elif isinstance(e, ast.GT):
            for x, y in zip(children, children[1:]):
                if not x > y:
                    return False
            return True

        elif isinstance(e, ast.LEQ):
            for x, y in zip(children, children[1:]):
                if not x <= y:
                    return False
            return True

        elif isinstance(e, ast.GEQ):
            for x, y in zip(children, children[1:]):
                if not x >= y:
                    return False
            return True

        elif isinstance(e, ast.EQ):
            for x in children:
                for y in children[1:]:
                    if not x == y:
                        return False
            return True

        elif isinstance(e, ast.NEQ):
            for x in children:
                for y in children[1:]:
                    if not x != y:
                        return False
            return True

        elif isinstance(e, ast.Expr):
            raise ValueError('unimplemented: {}'.format(repr(e)))

        else:
            raise ValueError('what is this: {}'.format(repr(e)))


from ..fpbench import fpcparser

fpc_minimal = fpcparser.compile(
"""(FPCore (a b) (- (+ a b) a))
""")[0]

fpc_example = fpcparser.compile(
"""(FPCore (a b c)
 :name "NMSE p42, positive"
 :cite (hamming-1987 herbie-2015)
 :fpbench-domain textbook
 :pre (and (>= (* b b) (* 4 (* a c))) (!= a 0))
 (/ (+ (- b) (sqrt (- (* b b) (* 4 (* a c))))) (* 2 a)))
""")[0]

fpc_fmod2pi = fpcparser.compile(
"""(FPCore ()
 (- (* 2 (+ (+ (* 4 7.8539812564849853515625e-01) (* 4 3.7748947079307981766760e-08)) (* 4 2.6951514290790594840552e-15)))
    (* 2 PI))
)
""")[0]

fpc_loop = fpcparser.compile(
"""(FPCore ()
 (while (< x 100) ([x 0 (+ x PI)]) x))
""")[0]
