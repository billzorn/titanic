# all the things you wish you didn't know about floating point

import typing

import antlr4
from FPCoreLexer import FPCoreLexer
from FPCoreParser import FPCoreParser
from FPCoreVisitor import FPCoreVisitor

import fpcast as ast


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
    'fmod' : _reserve('unimplemented: fmod'),
    'remainder' : _reserve('unimplemented: remainder'),
    # rounding and truncation
    'ceil' : _reserve('unimplemented: ceil'),
    'floor' : _reserve('unimplemented: floor'),
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
    'sin' : _reserve('unimplemented: sin'),
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
    'pow' : _reserve('unimplemented: pow'),
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
    def __init__(self, inputs, e, props={}, core_id=None):
        # cross-core ID is not represented yet
        self.inputs = inputs
        self.e = e
        self.props = props
        self.name = self.props.get(':name', None)
        self.pre = self.props.get(':pre', None)

    def __str__(self):
        return 'FPCore ({})\n  name: {}\n   pre: {}\n  {}'.format(
            ' '.join(self.inputs), self.name, self.pre, self.e)

    def __repr__(self):
        return 'FPCoreObject(\n  {},\n  {},\n  {}\n)'.format(
            repr(self.inputs), repr(self.props), repr(self.e))

    def _sexp(self):
        return '(FPCore ({}) {} {})'.format(
            ' '.join(self.inputs),
            ' '.join(name + ' ' + propToSexp(prop) for name, prop in self.props.items()),
            str(self.e))
    sexp = property(_sexp)


class Visitor(FPCoreVisitor):
    def visitParse(self, ctx) -> typing.List[FPCore]:
        return [x for x in (child.accept(self) for child in ctx.getChildren()) if x]

    def visitFpcore(self, ctx) -> FPCore:
        core_id = ctx.cid.text if ctx.cid is not None else None
        # need a real error strategy
        if core_id in reserved_constructs:
            raise ValueError('name {} is reserved'.format(core_id))

        inputs = [x.text for x in ctx.inputs]
        input_set = set()
        for x in inputs:
            if x in reserved_constants:
                raise ValueError('argument name {} is reserved'.format(x))
            elif x in input_set:
                raise ValueError('duplicate argument name {}'.format(x))
            else:
                input_set.add(x)

        props = {name: x for name, x in (prop.accept(self) for prop in ctx.props)}
        e = ctx.e.accept(self)
        return FPCore(inputs, e, props=props, core_id=core_id)

    def visitExprNumeric(self, ctx) -> ast.Expr:
        return ast.Val(ctx.n.text)

    def visitExprSymbolic(self, ctx) -> ast.Expr:
        x = ctx.x.text
        if x in reserved_constants:
            return reserved_constants[x]
        else:
            return ast.Var(ctx.x.text)

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

    def visitExprOp(self, ctx):
        op = ctx.op.text
        if op in reserved_constructs:
            return reserved_constructs[op](*(arg.accept(self) for arg in ctx.args))
        else:
            raise ValueError('unsupported: call to core operator {}'.format(op))

    def visitPropStr(self, ctx):
        return ctx.name.text, ctx.s.text[1:-1]

    def visitPropList(self, ctx):
        return ctx.name.text, [x.text for x in ctx.xs]

    def visitPropExpr(self, ctx):
        return ctx.name.text, ctx.e.accept(self)


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



if __name__ == '__main__':
    import sys

    core = compfile('minimal.fpcore')
    print(core)

    pie = '3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068'
    
    float_args = {
        'a' : float('1e12'),
        'b' : float(pie),
    }

    sink_args = {
        'a' : ast.Sink(ast.mpfr('1e12', 53), inexact=False),
        'b' : ast.Sink(ast.mpfr(pie, 53), inexact=True),
    }
    
    print(core.e.evaluate(ast.EvalCtx(
        float_args,
        w=11,
        p=53,
        mode=ast.EVAL_754)))

    print(core.e.evaluate_sink(ast.EvalCtx(
        sink_args,
        w=11,
        p=53,
        mode=ast.EVAL_754)))

    print(core.e.evaluate_sink(ast.EvalCtx(
        sink_args,
        w=11,
        p=53,
        mode=ast.EVAL_OPTIMISTIC)))


    print('\n\n')

    core = compfile('example.fpcore')
    print(core)

    float_args = {
        'a' : float('-1e8'),
        'b' : float('1.2'),
        'c' : float('0.5'),
    }
    
    sink_args = {
        'a' : ast.Sink(ast.mpfr('-1e8', 53), inexact=True),
        'b' : ast.Sink(ast.mpfr('1.2', 53), inexact=True),
        'c' : ast.Sink(ast.mpfr('0.5', 53), inexact=False),
    }
    
    print(core.e.evaluate(ast.EvalCtx(
        float_args,
        w=11,
        p=53,
        mode=ast.EVAL_754)))

    print(core.e.evaluate_sink(ast.EvalCtx(
        sink_args,
        w=11,
        p=53,
        mode=ast.EVAL_754)))

    print(core.e.evaluate_sink(ast.EvalCtx(
        sink_args,
        w=11,
        p=53,
        mode=ast.EVAL_OPTIMISTIC)))

    # print(results[0].e.evaluate_sink(ast.EvalCtx({"a":"1e16", "b":"3.14159"})))
