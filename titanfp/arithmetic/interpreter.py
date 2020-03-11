"""Base FPCore interpreter."""


import itertools

from ..titanic import utils, digital
from ..fpbench import fpcast as ast
from .evalctx import EvalCtx
from . import ndarray

class EvaluatorError(utils.TitanicError):
    """Base Titanic evaluator error."""

class EvaluatorUnimplementedError(EvaluatorError):
    """Unimplemented feature in Titanic evaluator."""

class EvaluatorUnboundError(EvaluatorError, LookupError):
    """Unbound variable encountered during evaluation."""


class Evaluator(object):
    """FPCore evaluator.
    Dispatches on type of expressions in the AST.
    """

    ctype = EvalCtx

    def _eval_expr(self, e, ctx):
        raise EvaluatorUnimplementedError('expr {}: unimplemented'.format(str(e)))

    def _eval_value(self, e, ctx):
        raise EvaluatorUnimplementedError('value {}: unimplemented'.format(str(e)))

    def _eval_var(self, e, ctx):
        raise EvaluatorUnimplementedError('var {}: unimplemented'.format(str(e)))

    def _eval_val(self, e, ctx):
        raise EvaluatorUnimplementedError('val {}: unimplemented'.format(str(e)))

    def _eval_ctx(self, e, ctx):
        # Note that let creates a new context, so the old one will
        # not be changed.
        return self.evaluate(e.body, ctx.let(props=e.props))

    def _eval_tensor(self, e, ctx):
        raise EvaluatorUnimplementedError('tensor {}: unimplemented'.format(str(e)))

    def _eval_if(self, e, ctx):
        raise EvaluatorUnimplementedError('control {}: unimplemented'.format(str(e)))

    def _eval_let(self, e, ctx):
        raise EvaluatorUnimplementedError('control {}: unimplemented'.format(str(e)))

    def _eval_while(self, e, ctx):
        raise EvaluatorUnimplementedError('control {}: unimplemented'.format(str(e)))

    def _eval_for(self, e, ctx):
        raise EvaluatorUnimplementedError('control {}: unimplemented'.format(str(e)))

    def _eval_op(self, e, ctx):
        raise EvaluatorUnimplementedError('op {}: unimplemented'.format(str(e)))

    def _eval_unknown(self, e, ctx):
        raise EvaluatorError('unknown operation {}'.format(e.name))

    _evaluator_dispatch = {
        # catch-all for otherwise unimplemented AST nodes
        ast.Expr: '_eval_expr',
        # the parent value class represents interned data that should be returned exactly
        ast.ValueExpr: '_eval_value',
        # structured data, which represents a tensor literal (unless it's malformed)
        ast.Data: '_eval_data',
        # variables are their own thing
        ast.Var: '_eval_var',
        # catch-all for all constant-valued expressions
        ast.Val: '_eval_val',
        # specific types of values
        ast.Constant: '_eval_constant',
        ast.Decnum: '_eval_decnum',
        ast.Hexnum: '_eval_hexnum',
        ast.Integer: '_eval_integer',
        ast.Rational: '_eval_rational',
        ast.Digits: '_eval_digits',
        # strings are special ValueExprs (not Vals) that won't normally be evaluated
        ast.String: '_eval_string',
        # rounding contexts
        ast.Ctx: '_eval_ctx',
        # Tensors
        ast.Tensor: '_eval_tensor',
        ast.TensorStar: '_eval_tensorstar',
        # control flow
        ast.If: '_eval_if',
        ast.Let: '_eval_let',
        ast.LetStar: '_eval_letstar',
        ast.While: '_eval_while',
        ast.WhileStar: '_eval_whilestar',
        ast.For: '_eval_for',
        ast.ForStar: '_eval_forstar',
        # catch-all for operations with some number of arguments
        ast.NaryExpr: '_eval_op',
        # catch-all for operations not recognized by the compiler
        ast.UnknownOperator: '_eval_unknown',
        # specific operations
        ast.Cast: '_eval_cast',
        ast.Dim: '_eval_dim',
        ast.Size: '_eval_size',
        ast.Ref: '_eval_ref',
        ast.Add: '_eval_add',
        ast.Sub: '_eval_sub',
        ast.Mul: '_eval_mul',
        ast.Div: '_eval_div',
        ast.Sqrt: '_eval_sqrt',
        ast.Fma: '_eval_fma',
        ast.Neg: '_eval_neg',
        ast.Copysign: '_eval_copysign',
        ast.Fabs: '_eval_fabs',
        ast.Fdim: '_eval_fdim',
        ast.Fmax: '_eval_fmax',
        ast.Fmin: '_eval_fmin',
        ast.Fmod: '_eval_fmod',
        ast.Remainder: '_eval_remainder',
        ast.Ceil: '_eval_ceil',
        ast.Floor: '_eval_floor',
        ast.Nearbyint: '_eval_nearbyint',
        ast.Round: '_eval_round',
        ast.Trunc: '_eval_trunc',
        ast.Acos: '_eval_acos',
        ast.Acosh: '_eval_acosh',
        ast.Asin: '_eval_asin',
        ast.Asinh: '_eval_asinh',
        ast.Atan: '_eval_atan',
        ast.Atan2: '_eval_atan2',
        ast.Atanh: '_eval_atanh',
        ast.Cos: '_eval_cos',
        ast.Cosh: '_eval_cosh',
        ast.Sin: '_eval_sin',
        ast.Sinh: '_eval_sinh',
        ast.Tan: '_eval_tan',
        ast.Tanh: '_eval_tanh',
        ast.Exp: '_eval_exp',
        ast.Exp2: '_eval_exp2',
        ast.Expm1: '_eval_exmp1',
        ast.Log: '_eval_log',
        ast.Log10: '_eval_log10',
        ast.Log1p: '_eval_log1p',
        ast.Log2: '_eval_log2',
        ast.Cbrt: '_eval_cbrt',
        ast.Hypot: '_eval_hypot',
        ast.Pow: '_eval_pow',
        ast.Erf: '_eval_erf',
        ast.Erfc: '_eval_erfc',
        ast.Lgamma: '_eval_lgamma',
        ast.Tgamma: '_eval_tgamma',
        ast.LT: '_eval_lt',
        ast.GT: '_eval_gt',
        ast.LEQ: '_eval_leq',
        ast.GEQ: '_eval_geq',
        ast.EQ: '_eval_eq',
        ast.NEQ: '_eval_neq',
        ast.Isfinite: '_eval_isfinite',
        ast.Isinf: '_eval_isinf',
        ast.Isnan: '_eval_isnan',
        ast.Isnormal: '_eval_isnormal',
        ast.Signbit: '_eval_signbit',
        ast.And: '_eval_and',
        ast.Or: '_eval_or',
        ast.Not: '_eval_not',
    }

    _evaluator_cache = utils.ImmutableDict()


    def __init__(self):
        self.evals = 0
        self.bits_computed = 0
        self.eval_map = {}

    
    def evaluate(self, e, ctx):
        try:
            method = self._evaluator_cache[type(e)]
        except KeyError:
            # initialize the cache for this class if it hasn't been initialized already
            if isinstance(self._evaluator_cache, utils.ImmutableDict):
                self._evaluator_cache = {}
            # walk up the mro and assign the evaluator for the first subtype to this type
            method = None
            ecls = type(e)
            for superclass in ecls.__mro__:
                method_name = self._evaluator_dispatch.get(superclass, None)
                if method_name is not None and hasattr(self, method_name):
                    method = getattr(self, method_name)
                    self._evaluator_cache[ecls] = method
                    break
            if method is None:
                raise EvaluatorError('Evaluator: unable to dispatch for expression {} with mro {}'
                                     .format(repr(e), repr(ecls.__mro__)))

        self.evals += 1
        if self.evals & 0xfffff == 0xfffff:
            print(',', end='', flush=True)


        result = method(e, ctx)
        # print(repr(method))
        # print(str(e))
        # print(str(ctx.bindings))
        # print(' ->', str(result))
        # print()

        if hasattr(result, 'ctx'):
            self.bits_computed += result.ctx.nbits

        if id(e) in self.eval_map:
            self.eval_map[id(e)] += 1
        else:
            self.eval_map[id(e)] = 1

        return result


class BaseInterpreter(Evaluator):
    """Base interpreter - only implements control flow."""

    # datatype conversion

    dtype = type(None)
    ctype = EvalCtx

    constants = {
        'TRUE': True,
        'FALSE': False,
    }

    def __init__(self):
        super().__init__()
        self.cores = {}

    def arg_to_digital(x, ctx):
        raise EvaluatorUnimplementedError('arg_to_digital({}): unimplemented'.format(repr(x)))

    def round_to_context(x, ctx):
        raise EvaluatorUnimplementedError('round_to_context({}): unimplemented'.format(repr(x)))


    # values

    def _eval_value(self, e, ctx):
        return e.value

    def _eval_var(self, e, ctx):
        try:
            return ctx.bindings[e.value]
        except KeyError as exn:
            raise EvaluatorUnboundError(exn.args[0])

    def _eval_val(self, e, ctx):
        return self.arg_to_digital(e.value, ctx)

    def _eval_constant(self, e, ctx):
        try:
            return self.constants[e.value]
        except KeyError as exn:
            raise EvaluatorUnimplementedError('unsupported constant {}'.format(repr(exn.args[0])))

    def _eval_data(self, e, ctx):
        try:
            data, shape = ndarray.flatten_shaped_list(e.as_list())
            rounded_data = [self.evaluate(d, ctx) for d in data]
            return ndarray.NDArray(shape, rounded_data)
        except Exception:
            traceback.print_exc()
            raise EvaluatorError('expecting a well-formed tensor literal, got:\n{}'.format(str(e)))

    # Tensors

    def _eval_dim(self, e, ctx):
        nd = self.evaluate(e.children[0], ctx)
        if not isinstance(nd, ndarray.NDArray):
            raise EvaluatorError('{} must be a tensor to get its dimension'.format(repr(nd)))
        #return digital.Digital(m=len(nd.shape), exp=0)
        return self.arg_to_digital(len(nd.shape), ctx=ctx)

    def _eval_size(self, e, ctx):
        nd = self.evaluate(e.children[0], ctx)
        if not isinstance(nd, ndarray.NDArray):
            raise EvaluatorError('{} must be a tensor to get the size of a dimension'.format(repr(nd)))
        idx = self.evaluate(e.children[1], ctx)
        if not idx.is_integer():
            raise EvaluatorError('computed shape index {} must be an integer'.format(repr(idx)))
        i = int(idx.m * (2**idx.exp))
        #return digital.Digital(m=nd.shape[i], exp=0)
        return self.arg_to_digital(nd.shape[i], ctx=ctx)

    def _eval_ref(self, e, ctx):
        nd = self.evaluate(e.children[0], ctx)
        if not isinstance(nd, ndarray.NDArray):
            raise EvaluatorError('{} must be a tensor to get an element'.format(repr(nd)))
        pos = []
        for child in e.children[1:]:
            idx = self.evaluate(child, ctx)
            if not idx.is_integer():
                raise EvaluatorError('computed index {} must be an integer'.format(repr(idx)))
            pos.append(int(idx.m * (2**idx.exp)))
        result = nd[pos]
        if result is None:
            raise EvaluatorError('index {} has not been computed'.format(repr(pos)))
        else:
            return result

    def _eval_tensor(self, e, ctx):
        shape = []
        names = []
        for name, expr in e.dim_bindings:
            size = self.evaluate(expr, ctx)
            if not size.is_integer():
                raise EvaluatorError('dimension size {} must be an integer'.format(repr(size)))
            shape.append(int(size))
            names.append(name)

        # TODO: should coordinates be rounded?
        data = [self.evaluate(e.body, ctx.let(bindings=[(name, self.arg_to_digital(i, ctx=ctx))
                                                        for name, i in zip(names, ndarray.position(shape, idx))]))
                for idx in range(ndarray.shape_size(shape))]
        return ndarray.NDArray(shape=shape, data=data)

    def _eval_tensorstar(self, e, ctx):
        shape = []
        names = []
        for name, expr in e.dim_bindings:
            size = self.evaluate(expr, ctx)
            if not size.is_integer():
                raise EvaluatorError('dimension size {} must be an integer'.format(repr(size)))
            shape.append(int(size))
            names.append(name)

        nd = ndarray.NDArray(shape=shape)

        if e.ident:
            ctx = ctx.let(bindings=[(e.ident, nd)])

        for name, init_expr, update_expr in e.while_bindings:
            new_binding = (name, self.evaluate(init_expr, ctx))
            ctx = ctx.let(bindings=[new_binding])

        for idx in range(ndarray.shape_size(shape)):
            # TODO: should coordinates be rounded?
            pos = ndarray.position(shape, idx)
            ctx = ctx.let(bindings=[(name, self.arg_to_digital(i, ctx=ctx))
                                    for name, i in zip(names, pos)])
            for name, init_expr, update_expr in e.while_bindings:
                new_binding = (name, self.evaluate(update_expr, ctx))
                ctx = ctx.let(bindings=[new_binding])
            nd[pos] = self.evaluate(e.body, ctx)

        return nd


    # control flow

    def _eval_if(self, e, ctx):
        if self.evaluate(e.cond, ctx):
            return self.evaluate(e.then_body, ctx)
        else:
            return self.evaluate(e.else_body, ctx)

    def _eval_let(self, e, ctx):
        bindings = [(name, self.evaluate(expr, ctx)) for name, expr in e.let_bindings]
        return self.evaluate(e.body, ctx.let(bindings=bindings))

    # def _eval_letstar(self, e, ctx):
    #     bindings = []
    #     for name, expr in e.let_bindings:
    #         local_ctx = ctx.let(bindings=bindings)
    #         bindings.append((name, self.evaluate(expr, local_ctx)))
    #     return self.evaluate(e.body, ctx.let(bindings=bindings))

    def _eval_letstar(self, e, ctx):
        for name, expr in e.let_bindings:
            new_binding = (name, self.evaluate(expr, ctx))
            ctx = ctx.let(bindings=[new_binding])
        return self.evaluate(e.body, ctx)

    def _eval_while(self, e, ctx):
        bindings = [(name, self.evaluate(init_expr, ctx)) for name, init_expr, update_expr in e.while_bindings]
        ctx = ctx.let(bindings=bindings)
        while self.evaluate(e.cond, ctx):
            bindings = [(name, self.evaluate(update_expr, ctx)) for name, init_expr, update_expr in e.while_bindings]
            ctx = ctx.let(bindings=bindings)
        return self.evaluate(e.body, ctx)

    def _eval_whilestar(self, e, ctx):
        for name, init_expr, update_expr in e.while_bindings:
            new_binding = (name, self.evaluate(init_expr, ctx))
            ctx = ctx.let(bindings=[new_binding])
        while self.evaluate(e.cond, ctx):
            for name, init_expr, update_expr in e.while_bindings:
                new_binding = (name, self.evaluate(update_expr, ctx))
                ctx = ctx.let(bindings=[new_binding])
        return self.evaluate(e.body, ctx)

    def _eval_for(self, e, ctx):
        shape = []
        names = []
        for name, expr in e.dim_bindings:
            size = self.evaluate(expr, ctx)
            if not size.is_integer():
                raise EvaluatorError('dimension size {} must be an integer'.format(repr(size)))
            shape.append(int(size))
            names.append(name)

        bindings = [(name, self.evaluate(init_expr, ctx)) for name, init_expr, update_expr in e.while_bindings]
        ctx = ctx.let(bindings=bindings)

        for idx in range(ndarray.shape_size(shape)):
            # TODO: should coordinates be rounded?
            ctx = ctx.let(bindings=[(name, self.arg_to_digital(i, ctx=ctx))
                                    for name, i in zip(names, ndarray.position(shape, idx))])
            bindings = [(name, self.evaluate(update_expr, ctx)) for name, init_expr, update_expr in e.while_bindings]
            ctx = ctx.let(bindings=bindings)

        return self.evaluate(e.body, ctx)

    def _eval_forstar(self, e, ctx):
        shape = []
        names = []
        for name, expr in e.dim_bindings:
            size = self.evaluate(expr, ctx)
            if not size.is_integer():
                raise EvaluatorError('dimension size {} must be an integer'.format(repr(size)))
            shape.append(int(size))
            names.append(name)

        for name, init_expr, update_expr in e.while_bindings:
            new_binding = (name, self.evaluate(init_expr, ctx))
            ctx = ctx.let(bindings=[new_binding])

        for idx in range(ndarray.shape_size(shape)):
            # TODO: should coordinates be rounded?
            ctx = ctx.let(bindings=[(name, self.arg_to_digital(i, ctx=ctx))
                                    for name, i in zip(names, ndarray.position(shape, idx))])
            for name, init_expr, update_expr in e.while_bindings:
                new_binding = (name, self.evaluate(update_expr, ctx))
                ctx = ctx.let(bindings=[new_binding])

        return self.evaluate(e.body, ctx)

    def _eval_unknown(self, e, ctx):
        ident = e.name
        if ident is not None and ident in self.cores:
            function_core = self.cores[ident]
        else:
            raise EvaluatorError('unknown function {}'.format(e.name))

        # wrap in Value, to avoid rounding the inputs again
        args = [ast.ValueExpr(self.evaluate(child, ctx)) for child in e.children]

        return self.interpret(function_core, args, ctx=ctx, override=False)

    # interpreter interface

    def arg_ctx(self, core, args, ctx=None, override=True):
        if len(core.inputs) != len(args):
            raise ValueError('incorrect number of arguments: got {}, expecting {} ({})'.format(
                len(args), len(core.inputs), ' '.join((name for name, props in core.inputs))))

        if ctx is None:
            ctx = self.ctype(props=core.props)
        elif override:
            allprops = {}
            allprops.update(core.props)
            allprops.update(ctx.props)
            ctx = ctx.let(props=allprops)
        else:
            ctx = ctx.let(props=core.props)

        arg_bindings = []

        for arg, (name, props, shape) in zip(args, core.inputs):
            local_ctx = ctx.let(props=props)

            if isinstance(arg, self.dtype):
                argval = self.round_to_context(arg, ctx=local_ctx)
            elif isinstance(arg, ast.Expr):
                argval = self.evaluate(arg, local_ctx)
            elif isinstance(arg, ndarray.NDArray):
                rounded_data = [
                    self.round_to_context(d, ctx=local_ctx) if isinstance(arg, self.dtype) else self.arg_to_digital(d, local_ctx)
                    for d in arg.data
                ]
                argval = ndarray.NDArray(shape=arg.shape, data=rounded_data)
            elif isinstance(arg, list):
                nd_unrounded = ndarray.NDArray(shape=None, data=arg)
                rounded_data = [
                    self.round_to_context(d, ctx=local_ctx) if isinstance(arg, self.dtype) else self.arg_to_digital(d, local_ctx)
                    for d in nd_unrounded.data
                ]
                argval = ndarray.NDArray(shape=nd_unrounded.shape, data=rounded_data)
            else:
                argval = self.arg_to_digital(arg, local_ctx)

            if isinstance(argval, ndarray.NDArray):
                if len(shape) != len(argval.shape):
                    raise EvaluatorError('tensor input has wrong shape: expecting {}, got {}'.format(repr(shape), repr(argval.shape)))
                for dim, argdim in zip(shape, argval.shape):
                    if isinstance(dim, int) and dim != argdim:
                        raise EvaluatorError('tensor input has wrong shape: expecting {}, got {}'.format(repr(shape), repr(argval.shape)))
                    elif isinstance(dim, str):
                        arg_bindings.append((dim, self.arg_to_digital(argdim, local_ctx)))

            arg_bindings.append((name, argval))

        return ctx.let(bindings=arg_bindings)

    def register_function(self, core):
        if core.ident is not None:
            self.cores[core.ident] = core

    def interpret(self, core, args, ctx=None, override=True):
        ctx = self.arg_ctx(core, args, ctx=ctx, override=override)
        return self.evaluate(core.e, ctx)

    def interpret_pre(self, core, args, ctx=None, override=True):
        if core.pre is None:
            return True
        else:
            ctx = self.arg_ctx(core, args, ctx=ctx, override=override)
            return self.evaluate(core.pre, ctx)


class SimpleInterpreter(BaseInterpreter):
    """Simple FPCore interpreter.
    Override Interpreter.dtype with any class that supports
    simple arithmetic operators: a + b, a > b, abs(a).
    """

    def _eval_cast(self, e, ctx):
        return self.round_to_context(self.evaluate(e.children[0], ctx), ctx)

    def _eval_add(self, e, ctx):
        return self.evaluate(e.children[0], ctx) + self.evaluate(e.children[1], ctx)

    def _eval_sub(self, e, ctx):
        return self.evaluate(e.children[0], ctx) - self.evaluate(e.children[1], ctx)

    def _eval_mul(self, e, ctx):
        return self.evaluate(e.children[0], ctx) * self.evaluate(e.children[1], ctx)

    # Division may need to be overwritten for types that raise an exception on
    # division by 0 (such as Python floats).

    def _eval_div(self, e, ctx):
        return self.evaluate(e.children[0], ctx) / self.evaluate(e.children[1], ctx)

    def _eval_neg(self, e, ctx):
        return -self.evaluate(e.children[0], ctx)

    def _eval_fabs(self, e, ctx):
        return abs(self.evaluate(e.children[0], ctx))

    # In the common case where nan isn't ordered, it will probably be necessary
    # to override these functions so that they do the right thing.

    def _eval_fmax(self, e, ctx):
        return max(self.evaluate(e.children[0], ctx), self.evaluate(e.children[1], ctx))

    def _eval_fmin(self, e, ctx):
        return min(self.evaluate(e.children[0], ctx), self.evaluate(e.children[1], ctx))

    # This ugly code is designed to cause comparison operators to short-circuit.
    # A more elegant implementation:
    #    children = [self.evaluate(child, ctx) for child in e.children]
    #    for x, y in zip(children, children[1:]):
    #        if not x < y:
    #            return False
    #    return True
    # does not short-circuit, which is inconsistent with the behavior of
    # _eval_and and _eval_or, which use the short-circuiting all and any builtins.

    def _eval_lt(self, e, ctx):
        if len(e.children) == 2:
            return self.evaluate(e.children[0], ctx) < self.evaluate(e.children[1], ctx)
        elif len(e.children) < 2:
            return True
        else:
            a = self.evaluate(e.children[0], ctx)
            for child in e.children[1:]:
                b = self.evaluate(child, ctx)
                if not a < b:
                    return False
                a = b
            return True

    def _eval_gt(self, e, ctx):
        if len(e.children) == 2:
            return self.evaluate(e.children[0], ctx) > self.evaluate(e.children[1], ctx)
        elif len(e.children) < 2:
            return True
        else:
            a = self.evaluate(e.children[0], ctx)
            for child in e.children[1:]:
                b = self.evaluate(child, ctx)
                if not a > b:
                    return False
                a = b
            return True

    def _eval_leq(self, e, ctx):
        if len(e.children) == 2:
            return self.evaluate(e.children[0], ctx) <= self.evaluate(e.children[1], ctx)
        elif len(e.children) < 2:
            return True
        else:
            a = self.evaluate(e.children[0], ctx)
            for child in e.children[1:]:
                b = self.evaluate(child, ctx)
                if not a <= b:
                    return False
                a = b
            return True

    def _eval_geq(self, e, ctx):
        if len(e.children) == 2:
            return self.evaluate(e.children[0], ctx) >= self.evaluate(e.children[1], ctx)
        elif len(e.children) < 2:
            return True
        else:
            a = self.evaluate(e.children[0], ctx)
            for child in e.children[1:]:
                b = self.evaluate(child, ctx)
                if not a >= b:
                    return False
                a = b
            return True

    def _eval_eq(self, e, ctx):
        if len(e.children) == 2:
            return self.evaluate(e.children[0], ctx) == self.evaluate(e.children[1], ctx)
        elif len(e.children) < 2:
            return True
        else:
            a = self.evaluate(e.children[0], ctx)
            for child in e.children[1:]:
                b = self.evaluate(child, ctx)
                if not a == b:
                    return False
                a = b
            return True

    # This is particularly complicated, because we need to try every combination
    # to ensure that each element is distinct. To short-circuit but avoid recomputation,
    # we keep a cache of results we've already computed. Note that we can't just
    # index with the child expressions themselves, or we might notice that shared
    # subtrees are actually the same object and forget to re-evaluate some of them.

    def _eval_neq(self, e, ctx):
        if len(e.children) == 2:
            return self.evaluate(e.children[0], ctx) != self.evaluate(e.children[1], ctx)
        elif len(e.children) < 2:
            return True
        else:
            nchildren = len(e.children)
            cached = [False] * nchildren
            cache = [None] * nchildren
            for i1, i2 in itertools.combinations(range(nchildren), 2):
                if cached[i1]:
                    a = cache[i1]
                else:
                    a = self.evaluate(e.children[i1], ctx)
                    cache[i1] = a
                    cached[i1] = True
                if cached[i2]:
                    b = cache[i2]
                else:
                    b = self.evaluate(e.children[i2], ctx)
                    cache[i2] = b
                    cached[i2] = True
                if not a != b:
                    return False
            return True

    def _eval_and(self, e, ctx):
        return all(self.evaluate(child, ctx) for child in e.children)

    def _eval_or(self, e, ctx):
        return any(self.evaluate(child, ctx) for child in e.children)

    def _eval_not(self, e, ctx):
        return not self.evaluate(e.children[0], ctx)


class StandardInterpreter(SimpleInterpreter):
    """Standard FPCore interpreter.
    Override Interpreter.dtype with a class derived from Digital
    all standard FPCore operations as methods: x.add(y), a.fma(y, z), etc.
    i.e. MPNum.
    """

    def _eval_add(self, e, ctx):
        return self.evaluate(e.children[0], ctx).add(self.evaluate(e.children[1], ctx), ctx=ctx)

    def _eval_sub(self, e, ctx):
        return self.evaluate(e.children[0], ctx).sub(self.evaluate(e.children[1], ctx), ctx=ctx)

    def _eval_mul(self, e, ctx):
        return self.evaluate(e.children[0], ctx).mul(self.evaluate(e.children[1], ctx), ctx=ctx)

    def _eval_div(self, e, ctx):
        return self.evaluate(e.children[0], ctx).div(self.evaluate(e.children[1], ctx), ctx=ctx)

    def _eval_sqrt(self, e, ctx):
        return self.evaluate(e.children[0], ctx).sqrt(ctx=ctx)

    def _eval_fma(self, e, ctx):
        return self.evaluate(e.children[0], ctx).fma(self.evaluate(e.children[1], ctx), self.evaluate(e.children[2], ctx), ctx=ctx)

    def _eval_neg(self, e, ctx):
        return self.evaluate(e.children[0], ctx).neg(ctx=ctx)

    def _eval_copysign(self, e, ctx):
        return self.evaluate(e.children[0], ctx).copysign(self.evaluate(e.children[1], ctx), ctx=ctx)

    def _eval_fabs(self, e, ctx):
        return self.evaluate(e.children[0], ctx).fabs(ctx=ctx)

    def _eval_fdim(self, e, ctx):
        return self.evaluate(e.children[0], ctx).fdim(self.evaluate(e.children[1], ctx), ctx=ctx)

    def _eval_fmax(self, e, ctx):
        return self.evaluate(e.children[0], ctx).fmax(self.evaluate(e.children[1], ctx), ctx=ctx)

    def _eval_fmin(self, e, ctx):
        return self.evaluate(e.children[0], ctx).fmin(self.evaluate(e.children[1], ctx), ctx=ctx)

    def _eval_fmod(self, e, ctx):
        return self.evaluate(e.children[0], ctx).fmod(self.evaluate(e.children[1], ctx), ctx=ctx)

    def _eval_remainder(self, e, ctx):
        return self.evaluate(e.children[0], ctx).remainder(self.evaluate(e.children[1], ctx), ctx=ctx)

    def _eval_ceil(self, e, ctx):
        return self.evaluate(e.children[0], ctx).ceil(ctx=ctx)

    def _eval_floor(self, e, ctx):
        return self.evaluate(e.children[0], ctx).floor(ctx=ctx)

    def _eval_nearbyint(self, e, ctx):
        return self.evaluate(e.children[0], ctx).nearbyint(ctx=ctx)

    def _eval_round(self, e, ctx):
        return self.evaluate(e.children[0], ctx).round(ctx=ctx)

    def _eval_trunc(self, e, ctx):
        return self.evaluate(e.children[0], ctx).trunc(ctx=ctx)

    def _eval_acos(self, e, ctx):
        return self.evaluate(e.children[0], ctx).acos(ctx=ctx)

    def _eval_acosh(self, e, ctx):
        return self.evaluate(e.children[0], ctx).acosh(ctx=ctx)

    def _eval_asin(self, e, ctx):
        return self.evaluate(e.children[0], ctx).asin(ctx=ctx)

    def _eval_asinh(self, e, ctx):
        return self.evaluate(e.children[0], ctx).asinh(ctx=ctx)

    def _eval_atan(self, e, ctx):
        return self.evaluate(e.children[0], ctx).atan(ctx=ctx)

    def _eval_atan2(self, e, ctx):
        return self.evaluate(e.children[0], ctx).atan2(self.evaluate(e.children[1], ctx), ctx=ctx)

    def _eval_atanh(self, e, ctx):
        return self.evaluate(e.children[0], ctx).atanh(ctx=ctx)

    def _eval_cos(self, e, ctx):
        return self.evaluate(e.children[0], ctx).cos(ctx=ctx)

    def _eval_cosh(self, e, ctx):
        return self.evaluate(e.children[0], ctx).cosh(ctx=ctx)

    def _eval_sin(self, e, ctx):
        return self.evaluate(e.children[0], ctx).sin(ctx=ctx)

    def _eval_sinh(self, e, ctx):
        return self.evaluate(e.children[0], ctx).sinh(ctx=ctx)

    def _eval_tan(self, e, ctx):
        return self.evaluate(e.children[0], ctx).tan(ctx=ctx)

    def _eval_tanh(self, e, ctx):
        return self.evaluate(e.children[0], ctx).tanh(ctx=ctx)

    def _eval_exp(self, e, ctx):
        return self.evaluate(e.children[0], ctx).exp_(ctx=ctx)

    def _eval_exp2(self, e, ctx):
        return self.evaluate(e.children[0], ctx).exp2(ctx=ctx)

    def _eval_expm1(self, e, ctx):
        return self.evaluate(e.children[0], ctx).expm1(ctx=ctx)

    def _eval_log(self, e, ctx):
        return self.evaluate(e.children[0], ctx).log(ctx=ctx)

    def _eval_log10(self, e, ctx):
        return self.evaluate(e.children[0], ctx).log10(ctx=ctx)

    def _eval_log1p(self, e, ctx):
        return self.evaluate(e.children[0], ctx).log1p(ctx=ctx)

    def _eval_log2(self, e, ctx):
        return self.evaluate(e.children[0], ctx).log2(ctx=ctx)

    def _eval_cbrt(self, e, ctx):
        return self.evaluate(e.children[0], ctx).cbrt(ctx=ctx)

    def _eval_hypot(self, e, ctx):
        return self.evaluate(e.children[0], ctx).hypot(self.evaluate(e.children[1], ctx), ctx=ctx)

    def _eval_pow(self, e, ctx):
        return self.evaluate(e.children[0], ctx).pow(self.evaluate(e.children[1], ctx), ctx=ctx)

    def _eval_erf(self, e, ctx):
        return self.evaluate(e.children[0], ctx).erf(ctx=ctx)

    def _eval_erfc(self, e, ctx):
        return self.evaluate(e.children[0], ctx).erfc(ctx=ctx)

    def _eval_lgamma(self, e, ctx):
        return self.evaluate(e.children[0], ctx).lgamma(ctx=ctx)

    def _eval_tgamma(self, e, ctx):
        return self.evaluate(e.children[0], ctx).tgamma(ctx=ctx)

    def _eval_isfinite(self, e, ctx):
        return self.evaluate(e.children[0], ctx).isfinite()

    def _eval_isinf(self, e, ctx):
        return self.evaluate(e.children[0], ctx).isinf

    def _eval_isnan(self, e, ctx):
        return self.evaluate(e.children[0], ctx).isnan

    def _eval_isnormal(self, e, ctx):
        return self.evaluate(e.children[0], ctx).isnormal()

    def _eval_signbit(self, e, ctx):
        return self.evaluate(e.children[0], ctx).signbit()
