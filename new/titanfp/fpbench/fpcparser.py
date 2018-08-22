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

    def visitNumber(self, ctx) -> ast.Expr:
        if ctx.n is not None:
            k = ctx.n.text
            if k in self._val_literals:
                return self._val_literals[k]
            else:
                return self._val_literals.setdefault(k, ast.Val(k))
        else:
            e = int(ctx.e.text)
            b = int(ctx.b.text)
            if b < 2:
                raise ValueError('base must be >= 2, got {}'.format(repr(b)))
            k = ctx.m.text, ctx.e.text, ctx.b.text
            if k in self._val_literals:
                return self._val_literals[k]
            else:
                return self._val_literals.setdefault(k, ast.Digits(*k))

    def visitExprNum(self, ctx) -> ast.Expr:
        return ctx.n.accept(self)

    def visitExprSym(self, ctx) -> ast.Expr:
        k = ctx.x.text
        if k in reserved_constants:
            return reserved_constants[k]
        elif k in self._var_literals:
            return self._var_literals[k]
        else:
            return self._var_literals.setdefault(k, ast.Var(k))

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
            raise ValueError('unsupported: call to FPCore operator {}'.format(op))

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
    fpc_minimal = """(FPCore () (+ 1 (digits 1 0 2)))
"""
    fpc_example = """(FPCore (a b c)
 :name "NMSE p42, positive"
 :cite (hamming-1987 herbie-2015)
 :fpbench-domain textbook
 :pre (and (>= (* b b) (* 4 (* a c))) (!= a 0))
 (/ (+ (- b) (sqrt (- (* b b) (* 4 (* a c))))) (* 2 a)))
"""
    fpc_big = """(FPCore ()
 :name arclength
 :precision binary128
 (let ([n (! :precision integer 1000000)]
       [dppi (! :precision PI PI)])
   (let ([h (! :precision binary64 (/ dppi n))])
     (while (<= i n)
      ([s1
        (! :precision binary64 0.0)
        (let ([t2 (let ([x (! :precision binary64 (* i h))])
                    ;; inlined body of fun
                    (while (<= k 5)
                     ([d0
                       (! :precision binary32 2.0)
                       (! :precision binary32 (* 2.0 d0))]
                      [t0
                       x
                       (! :precision binary64 (+ t0 (/ (sin (* d0 x)) d0)))]
                      [k 1 (+ k 1)])
                     t0))])
          (+ s1 (! :precision binary64 (sqrt (+ (* h h) (* (- t2 t1) (- t2 t1)))))))]
       [t1
        (! :precision binary64 0.0)
        (let ([t2 (let ([x (! :precision binary64 (* i h))])
                    ;; inlined body of fun
                    (while (<= k 5)
                     ([d0
                       (! :precision binary32 2.0)
                       (! :precision binary32 (* 2.0 d0))]
                      [t0
                       x
                       (! :precision binary64 (+ t0 (/ (sin (* d0 x)) d0)))]
                      [k 1 (+ k 1)])
                     t0))])
          t2)]
       [i
        (! :precision integer 1)
        (! :precision integer (+ i 1))])
      s1))))
"""

    core = compile(fpc_minimal)[0]
    print(core)
    print(core.sexp)
    print(repr(core))

    core = compile(fpc_example)[0]
    print(core)
    print(core.sexp)
    print(repr(core))

    core = compile(fpc_big)[0]
    print(core)
    print(core.sexp)
    print(repr(core))
