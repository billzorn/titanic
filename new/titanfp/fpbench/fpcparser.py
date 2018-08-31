import typing

import antlr4
from .FPCoreLexer import FPCoreLexer
from .FPCoreParser import FPCoreParser
from .FPCoreVisitor import FPCoreVisitor

from . import fpcast as ast

def _neg_or_sub(a, b=None):
    if b is None:
        return ast.Neg(a)
    else:
        return ast.Sub(a, b)

reserved_constructs = {
    # reserved
    'FPCore' : None,
    # control flow (these asts are assembled directly in the visitor)
    '!' : None,
    'cast' : None,
    'if' : None,
    'let' : None,
    'while' : None,
    'digits' : None,

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
    '1_PI' : ast.Constant('1_PI'),
    '2_PI' : ast.Constant('2_PI'),
    '2_SQRTPI' : ast.Constant('2_SQRTPI'),
    'SQRT2' : ast.Constant('SQRT2'),
    'SQRT1_2' : ast.Constant('SQRT1_2'),
    # infinity and NaN
    'INFINITY' : ast.Constant('inf'),
    'NAN' : ast.Constant('nan'),
    # boolean constants
    'TRUE' : ast.Constant('TRUE'),
    'FALSE' : ast.Constant('FALSE'),
}


class Visitor(FPCoreVisitor):
    def _parse_props(self, props):
        parsed = {}
        for prop in props:
            name, x = prop.accept(self)
            if name in parsed:
                raise ValueError('duplicate property {}'.format(name))
            parsed[name] = x
        return parsed

    def __init__(self):
        super().__init__()
        self._val_literals = {}
        self._var_literals = {}

    def visitParse(self, ctx) -> typing.List[ast.FPCore]:
        return [x for x in (child.accept(self) for child in ctx.getChildren()) if x]

    def visitFpcore(self, ctx) -> ast.FPCore:
        inputs = [arg.accept(self) for arg in ctx.inputs]

        input_set = set()
        for name, props in inputs:
            if name in input_set:
                raise ValueError('duplicate argument name {}'.format(name))
            # # Maybe this is a useful feature?
            # elif name in reserved_constants:
            #     raise ValueError('argument name {} is reserved'.format(name))
            else:
                input_set.add(name)

        props = self._parse_props(ctx.props)
        e = ctx.e.accept(self)

        return ast.FPCore(inputs, e, props=props)

    def visitArgument(self, ctx):
        return ctx.name.text, self._parse_props(ctx.props)

    def visitNumberDec(self, ctx) -> ast.Expr:
        k = ctx.n.text
        if k in self._val_literals:
            return self._val_literals[k]
        else:
            v = ast.Decnum(k)
            self._val_literals[k] = v
            return v

    def visitNumberHex(self, ctx) -> ast.Expr:
        k = ctx.n.text
        if k in self._val_literals:
            return self._val_literals[k]
        else:
            v = ast.Hexnum(k)
            self._val_literals[k] = v
            return v

    def visitNumberRational(self, ctx) -> ast.Expr:
        p, q = ctx.n.text.split('/')
        k = int(p), int(q)
        if k in self._val_literals:
            return self._val_literals[k]
        else:
            v = ast.Rational(*k)
            self._val_literals[k] = v
            return v

    def visitNumberDigits(self, ctx) -> ast.Expr:
        try:
            k = int(ctx.m.text), int(ctx.e.text), int(ctx.b.text)
        except ValueError:
            raise ValueError('digits: m, e, b must be integers, got {}, {}, {}'
                             .format(repr(ctx.m.text), repr(ctx.e.text), repr(ctx.b.text)))
        if k in self._val_literals:
            return self._val_literals[k]
        else:
            if k[2] < 2:
                raise ValueError('digits: base must be >= 2, got {}'.format(repr(ctx.b.text)))
            v = ast.Digits(*k)
            self._val_literals[k] = v
            return v

    def visitExprNum(self, ctx) -> ast.Expr:
        return ctx.n.accept(self)

    def visitExprSym(self, ctx) -> ast.Expr:
        k = ctx.x.text
        if k in reserved_constants:
            return reserved_constants[k]
        elif k in self._var_literals:
            return self._var_literals[k]
        else:
            v = ast.Var(k)
            self._var_literals[k] = v
            return v

    def visitExprCtx(self, ctx) -> ast.Expr:
        return ast.Ctx(
            self._parse_props(ctx.props),
            ctx.body.accept(self),
        )

    def visitExprCast(self, ctx) -> ast.Expr:
        return ast.Cast(
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

    def visitExprOp(self, ctx):
        op = ctx.op.text
        if op in reserved_constructs:
            return reserved_constructs[op](*(arg.accept(self) for arg in ctx.args))
        else:
            return ast.UnknownOperator(*(arg.accept(self) for arg in ctx.args),
                                       name=op)

    def visitPropStr(self, ctx):
        name = ctx.name.text
        if name.startswith(':'):
            return name[1:], ctx.s.text[1:-1]
        else:
            raise ValueError('bad keyword {} in FPCore property'.format(name))

    def visitPropList(self, ctx):
        name = ctx.name.text
        if name.startswith(':'):
            return name[1:], [x.text for x in ctx.xs]
        else:
            raise ValueError('bad keyword {} in FPCore property'.format(name))

    # # This seems like a good idea, but it means prop symbols will print back out as
    # # strings, as we have no way to distinguish them.
    # def visitPropSym(self, ctx):
    #     name = ctx.name.text
    #     if name.startswith(':'):
    #         return name[1:], ctx.s.text
    #     else:
    #         raise ValueError('bad keyword {} in FPCore property'.format(name))

    def visitPropExpr(self, ctx):
        name = ctx.name.text
        if name.startswith(':'):
            return name[1:], ctx.e.accept(self)
        else:
            raise ValueError('bad keyword {} in FPCore property'.format(name))

    def visitPropDatum(self, ctx):
        name = ctx.name.text
        if name.startswith(':'):
            return name[1:], ctx.d.accept(self)
        else:
            raise ValueError('bad keyword {} in FPCore property'.format(name))

    def visitDatum(self, ctx):
        return ast.Data(ctx.getText())


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
    tree = parser.parse()
    errors = err_listener.syntax_errors
    if len(errors) > 0:
        err_text = ''.join('  ' + str(err) for err in errors)
        raise ValueError('unable to parse FPCore:\n' + err_text)
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
