# all the things you wish you didn't know about floating point


import typing

import antlr4
from .FPCoreLexer import FPCoreLexer
from .FPCoreParser import FPCoreParser
from .FPCoreVisitor import FPCoreVisitor

from . import fpcast as ast


def _reserve(message):
    def raise_reserved(*args):
        raise ValueError(message)
    return raise_reserved

reserved_constructs = {
    # reserved
    'FPCore' : _reserve('reserved: FPCore'),
    # control flow (these asts are assembled directly in the visitor)
    'if' : _reserve('reserved: if'),
    'let' : _reserve('reserved: let'),
    'while' : _reserve('reserved: while'),

    # ieee754 required arithmetic (negation is a special case of subtraction)
    '+' : ast.Add,
    '-' : lambda a, b=None: ast.Sub(a, b) if b is not None else ast.Neg(a),
    '*' : ast.Mul,
    '/' : ast.Div,
    'sqrt' : ast.Sqrt,
    'fma' : _reserve('unimplemented: fma'),
    # discrete operations
    'copysign' : _reserve('unimplemented: copysign'),
    'fabs' : _reserve('unimplemented: fabs'),
    # composite arithmetic
    'fdim' : _reserve('unimplemented: fdim'),
    'fmax' : _reserve('unimplemented: fmax'),
    'fmin' : _reserve('unimplemented: fmin'),
    'fmod' : ast.Fmod,
    'remainder' : _reserve('unimplemented: remainder'),
    # rounding and truncation
    'ceil' : _reserve('unimplemented: ceil'),
    'floor' : ast.Floor,
    'nearbyint' : _reserve('unimplemented: nearbyint'),
    'round' : _reserve('unimplemented: round'),
    'trunc' : _reserve('unimplemented: trunc'),
    # trig
    'acos' : _reserve('unimplemented: acos'),
    'acosh' : _reserve('unimplemented: acosh'),
    'asin' : _reserve('unimplemented: asin'),
    'asinh' : _reserve('unimplemented: asinh'),
    'atan' : _reserve('unimplemented: atan'),
    'atan2' : _reserve('unimplemented: atan2'),
    'atanh' : _reserve('unimplemented: atanh'),
    'cos' : _reserve('unimplemented: cos'),
    'cosh' : _reserve('unimplemented: cosh'),
    'sin' : ast.Sin,
    'sinh' : _reserve('unimplemented: sinh'),
    'tan' : _reserve('unimplemented: tan'),
    'tanh' : _reserve('unimplemented: tanh'),
    # exponentials
    'exp' : _reserve('unimplemented: exp'),
    'exp2' : _reserve('unimplemented: exp2'),
    'expm1' : _reserve('unimplemented: expm1'),
    'log' : _reserve('unimplemented: log'),
    'log10' : _reserve('unimplemented: log10'),
    'log1p' : _reserve('unimplemented: log1p'),
    'log2' : _reserve('unimplemented: log2'),
    # powers
    'cbrt' : _reserve('unimplemented: cbrt'),
    'hypot' : _reserve('unimplemented: hypot'),
    'pow' : ast.Pow,
    # other
    'erf' : _reserve('unimplemented: erf'),
    'erfc' : _reserve('unimplemented: erfc'),
    'lgamma' : _reserve('unimplemented: lgamma'),
    'tgamma' : _reserve('unimplemented: tgamma'),

    # comparison
    '<' : ast.LT,
    '>' : ast.GT,
    '<=' : ast.LEQ,
    '>=' : ast.GEQ,
    '==' : ast.EQ,
    '!=' : ast.NEQ,
    # testing
    'isfinite' : _reserve('unimplemented: isfinite'),
    'isinf' : _reserve('unimplemented: isinf'),
    'isnan' : _reserve('unimplemented: isnan'),
    'isnormal' : _reserve('unimplemented: isnormal'),
    'signbit' : _reserve('unimplemented: signbit'),
    # logic
    'and' : ast.And,
    'or' : ast.Or,
    'not' : ast.Not,
}

reserved_constants = {
    # mathematical constants
    'E' : ast.Val('E'),
    'LOG2E' : ast.Val('LOG2E'),
    'LOG10E' : ast.Val('LOG10E'),
    'LN2' : ast.Val('LN2'),
    'LN10' : ast.Val('LN10'),
    'PI' : ast.Val('PI'),
    'PI_2' : ast.Val('PI_2'),
    'PI_4' : ast.Val('PI_4'),
    '1_PI' : ast.Val('1_PI'),
    '2_PI' : ast.Val('2_PI'),
    '2_SQRTPI' : ast.Val('2_SQRTPI'),
    'SQRT2' : ast.Val('SQRT2'),
    'SQRT1_2' : ast.Val('SQRT1_2'),
    # infinity and NaN
    'INFINITY' : ast.Val('inf'),
    'NAN' : ast.Val('nan'),
    # boolean constants
    'TRUE' : ast.Val('TRUE'),
    'FALSE' : ast.Val('FALSE'),
}


def propToSexp(p):
    if isinstance(p, str):
        return '"' + p + '"'
    elif isinstance(p, list):
        return '(' + ' '.join(p) + ')'
    else:
        return str(p)


class FPCore(object):
    def __init__(self, inputs, e, props = None):
        self.inputs = inputs
        self.e = e
        if props is None:
            self.props = {}
        else:
            self.props = props

        self.name = self.props.get('name', None)
        self.pre = self.props.get('pre', None)
        self.spec = self.props.get('spec', None)
        self.prec = self.props.get('prec', None)
        self.rm = self.props.get('rm', None)

    def __str__(self):
        return 'FPCore ({})\n  name: {}\n   pre: {}\n  {}'.format(
            ' '.join((ast.annotate(name, props) for name, props in self.inputs)),
            self.name, self.pre, self.e)

    def __repr__(self):
        return 'FPCore(\n  {},\n  {},\n  props={}\n)'.format(
            repr(self.inputs), repr(self.e), repr(self.props))

    @property
    def sexp(self):
        return '(FPCore ({}) {}{})'.format(
            ' '.join((ast.annotate(name, props) for name, props in self.inputs)),
            ''.join(':' + name + ' ' + propToSexp(prop) + ' ' for name, prop in self.props.items()),
            str(self.e))


class Visitor(FPCoreVisitor):
    def visitParse(self, ctx) -> typing.List[FPCore]:
        return [x for x in (child.accept(self) for child in ctx.getChildren()) if x]

    def visitFpcore(self, ctx) -> FPCore:
        inputs = [arg.accept(self) for arg in ctx.inputs]

        input_set = set()
        for name, props in inputs:
            if name in reserved_constants:
                raise ValueError('argument name {} is reserved'.format(name))
            elif name in input_set:
                raise ValueError('duplicate argument name {}'.format(name))
            else:
                input_set.add(name)

        props = {name: x for name, x in (prop.accept(self) for prop in ctx.props)}
        e = ctx.e.accept(self)
        return FPCore(inputs, e, props=props)

    def visitArgument(self, ctx):
        return ctx.name.text, {name : x for name, x in (prop.accept(self) for prop in ctx.props)}

    def visitRoundedNumber(self, ctx) -> ast.Expr:
        return ast.Val(ctx.n.text)

    def visitRoundedHexnum(self, ctx) -> ast.Expr:
        return ast.Val(ctx.n.text)

    def visitRoundedSymbolic(self, ctx) -> ast.Expr:
        x = ctx.x.text
        if x in reserved_constants:
            return reserved_constants[x]
        else:
            return ast.Var(ctx.x.text)

    def visitRoundedDigits(self, ctx) -> ast.Expr:
        # some crude validity checking; note this does not allow exponents like 1e3
        e = int(ctx.e.text)
        b = int(ctx.b.text)
        if b < 2:
            raise ValueError('base must be >= 2, got {}'.format(repr(b)))
        return ast.Digits(ctx.m.text, ctx.e.text, ctx.b.text)

    def visitRoundedOp(self, ctx):
        op = ctx.op.text
        if op in reserved_constructs:
            return reserved_constructs[op](*(arg.accept(self) for arg in ctx.args))
        else:
            raise ValueError('unsupported: call to FPCore operator {}'.format(op))

    def visitExprIf(self, ctx) -> ast.Expr:
        return ast.If(
            ctx.cond.accept(self),
            ctx.then_body.accept(self),
            ctx.else_body.accept(self)
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

    def visitExprExplicit(self, ctx) -> ast.Expr:
        props = {name: x for name, x in (prop.accept(self) for prop in ctx.props)}
        body = ctx.body.accept(self)
        body.props = props
        return body

    def visitExprImplicit(self, ctx) -> ast.Expr:
        return ctx.body.accept(self)

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


def parse(s):
    input_stream = antlr4.InputStream(s)
    lexer = FPCoreLexer(input_stream)
    token_stream = antlr4.CommonTokenStream(lexer)
    parser = FPCoreParser(token_stream)
    tree = parser.parse()
    return parser, tree

def compile(s):
    parser, tree = parse(s)
    visitor = Visitor()
    return visitor.visit(tree)

def compfile(fname):
    with open(fname, 'r') as f:
        results = compile(f.read())
    return results[0]


def demo():
    fpc_minimal = """(FPCore (a b) (- (+ a b) a))
"""
    fpc_example = """(FPCore (a b c)
 :name "NMSE p42, positive"
 :cite (hamming-1987 herbie-2015)
 :fpbench-domain textbook
 :pre (and (>= (* b b) (* 4 (* a c))) (!= a 0))
 (/ (+ (- b) (sqrt (- (* b b) (* 4 (* a c))))) (* 2 a)))
"""

    core = compile(fpc_minimal)[0]
    print(core)

    core = compile(fpc_example)[0]
    print(core)
