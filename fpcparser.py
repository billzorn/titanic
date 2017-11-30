import antlr4
from gen.FPCoreLexer import FPCoreLexer
from gen.FPCoreParser import FPCoreParser
from gen.FPCoreVisitor import FPCoreVisitor

import fpcast as ast
ops = ast.operations

def propToSexp(p):
    if isinstance(p, str):
        return '"' + p + '"'
    elif isinstance(p, list):
        return '(' + ' '.join(p) + ')'
    else:
        return str(p)

class FPCoreObject(object):
    def __init__(self, input_vars, props, e):
        self.args = input_vars
        self.properties = props
        self.name = self.properties.get(':name', None)
        self.pre = self.properties.get(':pre', None)
        self.e = e

    def __str__(self):
        return 'FPCore ({})\n  name: {}\n   pre: {}\n  {}'.format(
            ' '.join(self.args), self.name, self.pre, self.e)

    def __repr__(self):
        return 'FPCoreObject(\n  {},\n  {},\n  {}\n)'.format(
            repr(self.args), repr(self.properties), repr(self.e))

    def _sexp(self):
        return '(FPCore ({}) {} {})'.format(
            ' '.join(self.args),
            ' '.join(name + ' ' + propToSexp(prop) for name, prop in self.properties.items()),
            str(self.e))
    sexp = property(_sexp)

class Visitor(FPCoreVisitor):
    def visitParse(self, ctx):
        cores = []
        for child in ctx.getChildren():
            parsed = child.accept(self)
            if parsed:
                cores.append(parsed)
        return cores

    def visitFpcore(self, ctx):
        input_vars = [t.text for t in ctx.inputs]
        props = {name: x for name, x in (p.accept(self) for p in ctx.props)}
        e = ctx.e.accept(self)
        return FPCoreObject(input_vars, props, e)

    def visitFpimp(self, ctx):
        raise ValueError('unsupported: FPImp')

    def visitExprNum(self, ctx):
        #print('Num: ' + ctx.c.text)
        return ast.Val(ctx.c.text)

    def visitExprConst(self, ctx):
        return ast.Val(ctx.c.text) # this won't work with numpy...

    def visitExprVar(self, ctx):
        #print('Var: ' + ctx.x.text)
        return ast.Var(ctx.x.text)

    def visitExprUnop(self, ctx):
        op = ctx.op.getText()
        # special case unary negation
        if op == '-':
            return ops['neg'](ctx.arg0.accept(self))
        else:
            return ops[op](ctx.arg0.accept(self))

    def visitExprBinop(self, ctx):
        op = ctx.op.getText()
        return ops[op](ctx.arg0.accept(self), ctx.arg1.accept(self))

    def visitExprComp(self, ctx):
        op = ctx.op.getText()
        return ops[op](*(e.accept(self) for e in ctx.args))

    def visitExprLogical(self, ctx):
        op = ctx.op.getText()
        return ops[op](*(e.accept(self) for e in ctx.args))

    def visitExprIf(self, ctx):
        raise ValueError('unsupported: If')

    def visitExprLet(self, ctx):
        raise ValueError('unsupported: Let')

    def visitExprWhile(self, ctx):
        raise ValueError('unsupported: While')

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
        
