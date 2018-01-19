# all the things you wish you didn't know about floating point

import antlr4
from FPCoreLexer import FPCoreLexer
from FPCoreParser import FPCoreParser
from FPCoreVisitor import FPCoreVisitor

import fpcast as ast


reserved_constructs = {
    # reserved
    'FPCore' : None,
    # control flow
    'if' : ast.If,
    'let' : ast.Let,
    'while' : ast.While,
    
    # ieee754 required arithmetic
    '+' : ast.Add,
    '-' : ast.Sub, # unary negation is a special case
    '*' : ast.Mul,
    '/' : ast.Div,
    'sqrt' : ast.Sqrt,
    'fma' : None,
    # discrete operations
    'copysign' : None,
    'fabs' : None,
    # composite arithmetic
    'fdim' : None,
    'fmax' : None,
    'fmin' : None,
    'fmod' : None,
    'remainder' : None,
    # rounding and truncation
    'ceil' : None,
    'floor' : None,
    'nearbyint' : None,
    'round' : None,
    'trunc' : None,
    # trig
    'acos' : None,
    'acosh' : None,
    'asin' : None,
    'asinh' : None,
    'atan' : None,
    'atan2' : None,
    'atanh' : None,
    'cos' : None,
    'cosh' : None,
    'sin' : None,
    'sinh' : None,
    'tan' : None,
    'tanh' : None,
    # exponentials
    'exp' : None,
    'exp2' : None,
    'expm1' : None,
    'log' : None,
    'log10' : None,
    'log1p' : None,
    'log2' : None,
    # powers
    'cbrt' : None,
    'hypot' : None,
    'pow' : None,
    # other
    'erf' : None,
    'erfc' : None,
    'lgamma' : None,
    'tgamma' : None,

    # comparison
    '<' : ast.LT,
    '>' : ast.GT,
    '<=' : ast.LEQ,
    '>=' : ast.GEQ,
    '==' : ast.EQ,
    '!=' : ast.NEQ,
    # testing
    'isfinite' : None,
    'isinf' : None,
    'isnan' : None,
    'isnormal' : None,
    'signbit' : None,
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
            ' '.join(self.args), self.name, self.pre, self.e)

    def __repr__(self):
        return 'FPCoreObject(\n  {},\n  {},\n  {}\n)'.format(
            repr(self.args), repr(self.props), repr(self.e))

    def _sexp(self):
        return '(FPCore ({}) {} {})'.format(
            ' '.join(self.args),
            ' '.join(name + ' ' + propToSexp(prop) for name, prop in self.properties.items()),
            str(self.e))
    sexp = property(_sexp)

    
class Visitor(FPCoreVisitor):
    def visitParse(self, ctx) -> typing.List[FPCore]:
        return [x for x in (child.accept(self) for child in ctx.getChildren()) if x]

    def visitFpcore(self, ctx) -> FPCore:
        core_id = ctx.cid.text
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
        return FPCoreObject(inputs, e, props=props, core_id=core_id)

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
            [*zip(x.text for x in ctx.xs, e.accept(self) for e in ctx.es)],
            ctx.body.accept(self),
        )

    def visitExprWhile(self, ctx) -> ast.Expr:
        return ast.While(
            ctx.cond.accept(self),
            [*zip(
                x.text for x in ctx.xs,
                e0.accept(self) for e0 in ctx.e0s,
                e.accept(self) for e in ctx.es,
            )],
            ctx.body.accept(self),
        )

    def visitExprOp(self, ctx):
        #TODOTODOTODO
        op = ctx.op.getText()
        # special case unary negation
        if op == '-':
            return ops['neg'](ctx.arg0.accept(self))
        else:
            return ops[op](ctx.arg0.accept(self))



    def visitPropStr(self, ctx):
        return ctx.name.text, ctx.s.text[1:-1]

    def visitPropList(self, ctx):
        return ctx.name.text, [t.text for t in ctx.syms]

    def visitPropExpr(self, ctx):
        return ctx.name.text, ctx.e.accept(self)

def parse(s):
    input_stream = antlr4.InputStream(s)
    lexer = FPCoreLexer(input_stream)
    token_stream = antlr4.CommonTokenStream(lexer)
    parser = FPCoreParser(token_stream)
    parse_tree = parser.parse()
    return parser, parse_tree

def compile(s):
    parser, tree = parse(s)
    visitor = Visitor()
    return visitor.visit(tree)

if __name__ == '__main__':
    import sys
    parser, tree = parse(sys.stdin.read())

    visitor = Visitor()
    results = visitor.visit(tree)
    for x in results:
        print(str(x))
        print()
        print(repr(x))
        print()
        print(x.sexp)

    #print(results[0].e({'x': 0.125}));
