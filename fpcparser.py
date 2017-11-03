import antlr4
from gen.FPCoreLexer import FPCoreLexer
from gen.FPCoreParser import FPCoreParser
from gen.FPCoreVisitor import FPCoreVisitor

import fpcast as ast

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
        return '(FPCore ({}) {} {})'.format(
            ' '.join(self.args),
            ' '.join(name + ' ' + repr(prop) for name, prop in self.properties.items()),
            repr(self.e))

def impl_by_pairs(op, conj):
    def impl(*args):
        if len(args) > 2:
            return conj(*(op(args[i], args[i+1]) for i in range(len(args)-1)))
        else:
            return op(*args)
    return impl

def impl_all_pairs(op, conj):
    def impl(*args):
        if len(args) > 2:
            return conj(*(op(args[i], args[j]) for i in range(len(args)-1) for j in range(i+1,len(args))))
        else:
            return op(*args)
    return impl

operations = {
    ast.Add.name : ast.Add,
    ast.Sub.name : ast.Sub,
    ast.Mul.name : ast.Mul,
    ast.Div.name : ast.Div,
    ast.Sqrt.name : ast.Sqrt,
    ast.Neg.name : ast.Neg,
    ast.LT.name : impl_by_pairs(ast.LT, ast.And),
    ast.GT.name : impl_by_pairs(ast.GT, ast.And),
    ast.LEQ.name : impl_by_pairs(ast.LEQ, ast.And),
    ast.GEQ.name : impl_by_pairs(ast.GEQ, ast.And),
    ast.EQ.name : impl_by_pairs(ast.EQ, ast.And),
    ast.NEQ.name : impl_all_pairs(ast.NEQ, ast.And),
    ast.And.name : ast.And,
    ast.Or.name : ast.Or,
    ast.Not.name : ast.Not,
}

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
        return ast.Val(ctx.c.text)

    def visitExprConst(self, ctx):
        return ast.Val(ctx.c.text) # this won't work with numpy...

    def visitExprVar(self, ctx):
        return ast.Var(ctx.x.text)

    def visitExprOp(self, ctx):
        op = ctx.op.getText()
        # special case unary negation
        if op == '-' and len(ctx.args) == 1:
            return ast.Neg(*(e.accept(self) for e in ctx.args))
        else:
            return operations[op](*(e.accept(self) for e in ctx.args))

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
