import typing
import re

import antlr4
from .FPCoreLexer import FPCoreLexer
from .FPCoreParser import FPCoreParser
from .FPCoreVisitor import FPCoreVisitor

from . import fpcast as ast


_int_re = re.compile(r'(?P<sgn>[-+]?)(?:(?P<dec>[0-9]+)|(?P<hex>0[xX][0-9A-Fa-f]+))')

def read_int(s):
    m = _int_re.fullmatch(s)
    if m:
        if m.group('dec'):
            return int(s, 10)
        else: # m.group('hex')
            return int(s, 16)
    else:
        return None


def _neg_or_sub(a, b=None):
    if b is None:
        return ast.Neg(a)
    else:
        return ast.Sub(a, b)

reserved_constructs = {
    # reserved
    'FPCore' : None,
    # annotations and special syntax
    '!' : None,
    '#' : None,
    'cast' : ast.Cast,
    'digits' : None,
    # control flow (these asts are assembled directly in the visitor)
    'if' : None,
    'let' : None,
    'let*' : None,
    'while' : None,
    'while*' : None,
    'for' : None,
    'for*' : None,
    'tensor' : None,
    'tensor*' : None,

    # tensor operations
    'dim' : ast.Dim,
    'size' : ast.Size,
    'ref' : ast.Ref,
    # IEEE 754 required arithmetic (negation is a special case of subtraction)
    '+' : ast.Add,
    '-' : _neg_or_sub,
    '*' : ast.Mul,
    '/' : ast.Div,
    'sqrt' : ast.Sqrt,
    'fma' : ast.Fma,
    # discrete operations
    'copysign' : ast.Copysign,
    'fabs' : ast.Fabs,
    # composite arithmetic
    'fdim' : ast.Fdim,
    'fmax' : ast.Fmax,
    'fmin' : ast.Fmin,
    'fmod' : ast.Fmod,
    'remainder' : ast.Remainder,
    # rounding and truncation
    'ceil' : ast.Ceil,
    'floor' : ast.Floor,
    'nearbyint' : ast.Nearbyint,
    'round' : ast.Round,
    'trunc' : ast.Trunc,
    # trig
    'acos' : ast.Acos,
    'acosh' : ast.Acosh,
    'asin' : ast.Asin,
    'asinh' : ast.Asinh,
    'atan' : ast.Atan,
    'atan2' : ast.Atan2,
    'atanh' : ast.Atanh,
    'cos' : ast.Cos,
    'cosh' : ast.Cosh,
    'sin' : ast.Sin,
    'sinh' : ast.Sinh,
    'tan' : ast.Tan,
    'tanh' : ast.Tanh,
    # exponentials
    'exp' : ast.Exp,
    'exp2' : ast.Exp2,
    'expm1' : ast.Expm1,
    'log' : ast.Log,
    'log10' : ast.Log10,
    'log1p' : ast.Log1p,
    'log2' : ast.Log2,
    # powers
    'cbrt' : ast.Cbrt,
    'hypot' : ast.Hypot,
    'pow' : ast.Pow,
    # other
    'erf' : ast.Erf,
    'erfc' : ast.Erfc,
    'lgamma' : ast.Lgamma,
    'tgamma' : ast.Tgamma,

    # comparison
    '<' : ast.LT,
    '>' : ast.GT,
    '<=' : ast.LEQ,
    '>=' : ast.GEQ,
    '==' : ast.EQ,
    '!=' : ast.NEQ,
    # testing
    'isfinite' : ast.Isfinite,
    'isinf' : ast.Isinf,
    'isnan' : ast.Isnan,
    'isnormal' : ast.Isnormal,
    'signbit' : ast.Signbit,
    # logic
    'and' : ast.And,
    'or' : ast.Or,
    'not' : ast.Not,
}

reserved_constants = {
    # mathematical constants
    'E' : ast.Constant('E'),
    'LOG2E' : ast.Constant('LOG2E'),
    'LOG10E' : ast.Constant('LOG10E'),
    'LN2' : ast.Constant('LN2'),
    'LN10' : ast.Constant('LN10'),
    'PI' : ast.Constant('PI'),
    'PI_2' : ast.Constant('PI_2'),
    'PI_4' : ast.Constant('PI_4'),
    'M_1_PI' : ast.Constant('M_1_PI'),
    'M_2_PI' : ast.Constant('M_2_PI'),
    'M_2_SQRTPI' : ast.Constant('M_2_SQRTPI'),
    'SQRT2' : ast.Constant('SQRT2'),
    'SQRT1_2' : ast.Constant('SQRT1_2'),
    # infinity and NaN
    'INFINITY' : ast.Constant('INFINITY'),
    'NAN' : ast.Constant('NAN'),
    # boolean constants
    'TRUE' : ast.Constant('TRUE'),
    'FALSE' : ast.Constant('FALSE'),
}


class FPCoreParserError(Exception):
    """Unable to parse FPCore."""


class Visitor(FPCoreVisitor):
    def _parse_props(self, props):
        parsed = {}
        for prop in props:
            name, x = prop.accept(self)
            if name in parsed:
                raise FPCoreParserError('duplicate property {}'.format(name))
            parsed[name] = x
        return parsed

    def _parse_shape(self, shape):
        if shape:
            return [dim.accept(self) for dim in shape]
        else:
            return None

    def _intern_simple_number(self, k, cls):
        if k in self._num_literals:
            return self._num_literals[k]
        else:
            i = read_int(k)
            if i is not None:
                v = ast.Integer(i)
            else:
                v = cls(k)
            self._num_literals[k] = v
            return v

    def _intern_symbol(self, k):
        if k in self._sym_literals:
            return self._sym_literals[k]
        else:
            if k in reserved_constants:
                v = reserved_constants[k]
            else:
                v = ast.Var(k)
            self._sym_literals[k] = v
            return v

    def _intern_string(self, k):
        if k in self._str_literals:
            return self._str_literals[k]
        else:
            v = ast.String(k)
            self._str_literals[k] = v
            return v

    def __init__(self):
        super().__init__()
        self._num_literals = {}
        self._sym_literals = {}
        self._str_literals = {}

    def visitParse_fpcore(self, ctx) -> typing.List[ast.FPCore]:
        return [x for x in (child.accept(self) for child in ctx.getChildren()) if x]

    def visitParse_exprs(self, ctx) -> typing.List[ast.Expr]:
        return [x for x in (child.accept(self) for child in ctx.getChildren()) if x]

    def visitFpcore(self, ctx) -> ast.FPCore:
        if ctx.ident is None:
            ident = None
        else:
            ident = ctx.ident.text

        inputs = [arg.accept(self) for arg in ctx.inputs]

        input_set = set()
        for name, props, shape in inputs:
            if name in input_set:
                raise FPCoreParserError('duplicate argument name {}'.format(name))
            # Arguments that are shadowed by constants will never be able to be
            # referred to in the body, so we prevent that here.
            elif name in reserved_constants:
                raise FPCoreParserError('argument name {} is a reserved constant'.format(name))
            else:
                input_set.add(name)
            # Also check the names of dimensions.
            if shape:
                for dim in shape:
                    if isinstance(dim, str):
                        if dim in input_set:
                            raise FPCoreParserError('duplicate argument name {} for tensor dimension'.format(dim))
                        elif dim in reserved_constants:
                            raise FPCoreParserError('dimension name {} is a reserved constant'.format(dim))
                        else:
                            input_set.add(dim)


        props = self._parse_props(ctx.props)
        e = ctx.e.accept(self)
        name = props.get('name', None)
        if 'pre' in props:
            try:
                pre = data_as_expr(props['pre'], strict=True)
            except FPCoreParserError as e:
                raise FPCoreParserError('invalid precondition: ' + str(e))
        else:
            pre = None
        if 'spec' in props:
            try:
                spec = data_as_expr(props['spec'], strict=True)
            except FPCoreParserError as e:
                raise FPCoreParserError('invalid spec: ' + str(e))
        else:
            spec= None

        return ast.FPCore(inputs, e, props=props, ident=ident, name=name, pre=pre, spec=spec)

    def visitDimSym(self, ctx):
        return ctx.name.text

    def visitDimSize(self, ctx):
        n = ctx.size.accept(self)
        if isinstance(n, ast.Integer):
            return n.i
        else:
            raise FPCoreParserError('fixed dimension {} must be an integer'.format(ctx.size.text))

    def visitArgument(self, ctx):
        return ctx.name.text, self._parse_props(ctx.props), self._parse_shape(ctx.shape)

    def visitNumberDec(self, ctx) -> ast.Expr:
        return self._intern_simple_number(ctx.n.text, ast.Decnum)

    def visitNumberHex(self, ctx) -> ast.Expr:
        return self._intern_simple_number(ctx.n.text, ast.Hexnum)

    def visitNumberRational(self, ctx) -> ast.Expr:
        p, q = ctx.n.text.split('/')
        k = int(p), int(q)
        if k in self._num_literals:
            return self._num_literals[k]
        else:
            v = ast.Rational(*k)
            self._num_literals[k] = v
            return v

    def visitNumberDigits(self, ctx) -> ast.Expr:
        try:
            k = int(ctx.m.text), int(ctx.e.text), int(ctx.b.text)
        except ValueError:
            raise FPCoreParserError('digits: m, e, b must be integers, got {}, {}, {}'
                             .format(repr(ctx.m.text), repr(ctx.e.text), repr(ctx.b.text)))
        if k in self._num_literals:
            return self._num_literals[k]
        else:
            if k[2] < 2:
                raise FPCoreParserError('digits: base must be >= 2, got {}'.format(repr(ctx.b.text)))
            v = ast.Digits(*k)
            self._num_literals[k] = v
            return v

    def visitExprNum(self, ctx) -> ast.Expr:
        return ctx.n.accept(self)

    def visitExprSym(self, ctx) -> ast.Expr:
        return self._intern_symbol(ctx.x.text)

    def visitExprCtx(self, ctx) -> ast.Expr:
        return ast.Ctx(
            self._parse_props(ctx.props),
            ctx.body.accept(self),
        )

    def visitExprTensor(self, ctx) -> ast.Expr:
        return ast.Tensor(
            [*zip((x.text for x in ctx.xs), (e.accept(self) for e in ctx.es))],

            ctx.body.accept(self),
        )

    def visitExprTensorStar(self, ctx) -> ast.Expr:
        if ctx.name is None:
            ident = ''
        else:
            ident = ctx.name.text
        return ast.TensorStar(
            ident,
            [*zip((x.text for x in ctx.xs), (e.accept(self) for e in ctx.es))],
            [*zip(
                (x.text for x in ctx.while_xs),
                (e0.accept(self) for e0 in ctx.while_e0s),
                (e.accept(self) for e in ctx.while_es),
            )],
            ctx.body.accept(self),
        )

    def visitExprIf(self, ctx) -> ast.Expr:
        return ast.If(
            ctx.cond.accept(self),
            ctx.then_body.accept(self),
            ctx.else_body.accept(self),
        )

    def visitExprLet(self, ctx) -> ast.Expr:
        return ast.Let(
            [*zip((x.text for x in ctx.xs), (e.accept(self) for e in ctx.es))],
            ctx.body.accept(self),
        )

    def visitExprLetStar(self, ctx) -> ast.Expr:
        return ast.LetStar(
            [*zip((x.text for x in ctx.xs), (e.accept(self) for e in ctx.es))],
            ctx.body.accept(self),
        )

    def visitExprWhile(self, ctx) -> ast.Expr:
        return ast.While(
            ctx.cond.accept(self),
            [*zip(
                (x.text for x in ctx.xs),
                (e0.accept(self) for e0 in ctx.e0s),
                (e.accept(self) for e in ctx.es),
            )],
            ctx.body.accept(self),
        )

    def visitExprWhileStar(self, ctx) -> ast.Expr:
        return ast.WhileStar(
            ctx.cond.accept(self),
            [*zip(
                (x.text for x in ctx.xs),
                (e0.accept(self) for e0 in ctx.e0s),
                (e.accept(self) for e in ctx.es),
            )],
            ctx.body.accept(self),
        )

    def visitExprFor(self, ctx) -> ast.Expr:
        return ast.For(
            [*zip((x.text for x in ctx.xs), (e.accept(self) for e in ctx.es))],
            [*zip(
                (x.text for x in ctx.while_xs),
                (e0.accept(self) for e0 in ctx.while_e0s),
                (e.accept(self) for e in ctx.while_es),
            )],
            ctx.body.accept(self),
        )

    def visitExprForStar(self, ctx) -> ast.Expr:
        return ast.ForStar(
            [*zip((x.text for x in ctx.xs), (e.accept(self) for e in ctx.es))],
            [*zip(
                (x.text for x in ctx.while_xs),
                (e0.accept(self) for e0 in ctx.while_e0s),
                (e.accept(self) for e in ctx.while_es),
            )],
            ctx.body.accept(self),
        )

    def visitExprData(self, ctx) -> ast.Expr:
        return ast.TensorLit(ctx.d.accept(self))

    def visitExprSugarInt(self, ctx) -> ast.Expr:
        return ast.Ctx(
            {'precision': ast.Data(self._intern_symbol('integer'))},
            ctx.body.accept(self),
        )

    def visitExprOp(self, ctx) -> ast.Expr:
        op = ctx.op.text
        if op in reserved_constructs:
            return reserved_constructs[op](*(arg.accept(self) for arg in ctx.args))
        else:
            return ast.UnknownOperator(*(arg.accept(self) for arg in ctx.args),
                                       name=op)

    def visitProp(self, ctx) -> typing.Tuple[str, ast.Data]:
        name = ctx.name.text
        if name.startswith(':'):
            return name[1:], ast.Data(ctx.d.accept(self))
        else:
            raise FPCoreParserError('invalid keyword {} in FPCore property'.format(name))

    def visitDatumNum(self, ctx) -> ast.Expr:
        return ctx.n.accept(self)

    def visitDatumSym(self, ctx) -> ast.Expr:
        return self._intern_symbol(ctx.x.text)

    def visitDatumStr(self, ctx) -> ast.Expr:
        return self._intern_symbol(ctx.s.text[1:-1])

    def visitDatumList(self, ctx) -> typing.Tuple[ast.Expr]:
        return tuple(d.accept(self) for d in ctx.data)


class LogErrorListener(antlr4.error.ErrorListener.ErrorListener):

    def __init__(self):
        super().__init__()
        self.syntax_errors = []

    def syntaxError(self, recognizer, offending_symbol, line, column, msg, e):
        self.syntax_errors.append("line " + str(line) + ":" + str(column) + " " + msg)


def parse(s):
    input_stream = antlr4.InputStream(s)
    lexer = FPCoreLexer(input_stream)
    token_stream = antlr4.CommonTokenStream(lexer)
    parser = FPCoreParser(token_stream)
    err_listener = LogErrorListener()
    parser.removeErrorListeners()
    parser.addErrorListener(err_listener)
    tree = parser.parse_fpcore()
    errors = err_listener.syntax_errors
    if len(errors) > 0:
        err_text = ''.join('  ' + str(err) for err in errors)
        raise FPCoreParserError('unable to parse FPCore:\n' + err_text)
    else:
        return tree

def parse_exprs(s):
    input_stream = antlr4.InputStream(s)
    lexer = FPCoreLexer(input_stream)
    token_stream = antlr4.CommonTokenStream(lexer)
    parser = FPCoreParser(token_stream)
    err_listener = LogErrorListener()
    parser.removeErrorListeners()
    parser.addErrorListener(err_listener)
    tree = parser.parse_exprs()
    errors = err_listener.syntax_errors
    if len(errors) > 0:
        err_text = ''.join('  ' + str(err) for err in errors)
        raise FPCoreParserError('unable to parse expression:\n' + err_text)
    else:
        return tree


def compile(s):
    tree = parse(s)
    visitor = Visitor()
    return visitor.visit(tree)

def compile1(s):
    cores = compile(s)
    if len(cores) > 0:
        return cores[0]
    else:
        return None

def compfile(fname):
    with open(fname, 'rt') as f:
        cores = compile(f.read())
    return cores

def compfile1(fname):
    with open(fmane, 'rt') as f:
        cores = compile(f.read())
    if len(cores) > 0:
        return cores[0]
    else:
        return None

def read_exprs(s):
    tree = parse_exprs(s)
    visitor = Visitor()
    return visitor.visit(tree)

def data_as_expr(d, strict=False):
    if d.as_string() is None:
        try:
            es = read_exprs(str(d))
        except FPCoreParserError:
            if strict:
                raise
            else:
                return None
        if len(es) == 1:
            return es[0]
    # all other cases
    if strict:
        raise FPCoreParserError('data is not exactly one expression')
    else:
        return None
