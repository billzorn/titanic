import typing
import re

import antlr4
from .FPCoreLexer import FPCoreLexer
from .FPCoreParser import FPCoreParser
from .FPCoreVisitor import FPCoreVisitor

from .fpcommon import *
from . import fpcast as ast


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
        if self.intern_values:
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
        else:
            i = read_int(k)
            if i is not None:
                return ast.Integer(i)
            else:
                return cls(k)

    def _intern_rational(self, k):
        if self.intern_values:
            if k in self._num_literals:
                return self._num_literals[k]
            else:
                v = ast.Rational(*k)
                self._num_literals[k] = v
                return v
        else:
            return ast.Rational(*k)

    def _intern_digits(self, k):
        if self.intern_values:
            if k in self._num_literals:
                return self._num_literals[k]
            else:
                if k[2] < 2:
                    raise FPCoreParserError('digits: base must be >= 2, got {}'.format(repr(ctx.b.text)))
                v = ast.Digits(*k)
                self._num_literals[k] = v
                return v
        else:
            if k[2] < 2:
                raise FPCoreParserError('digits: base must be >= 2, got {}'.format(repr(ctx.b.text)))
            return ast.Digits(*k)

    def _intern_symbol(self, k):
        if self.intern_values:
            if k in self._sym_literals:
                return self._sym_literals[k]
            else:
                if k in reserved_constants:
                    v = reserved_constants[k]
                else:
                    v = ast.Var(k)
                self._sym_literals[k] = v
                return v
        else:
            if k in reserved_constants:
                return ast.Constant(k)
            else:
                return ast.Var(k)

    def _intern_string(self, k):
        if self.intern_values:
            if k in self._str_literals:
                return self._str_literals[k]
            else:
                v = ast.String(k)
                self._str_literals[k] = v
                return v
        else:
            return ast.String(k)

    def __init__(self):
        super().__init__()
        self.intern_values = False
        self._num_literals = {}
        self._sym_literals = {}
        self._str_literals = {}

    def visitParse_fpcore(self, ctx) -> typing.List[ast.FPCore]:
        return [x for x in (child.accept(self) for child in ctx.getChildren()) if x]

    def visitParse_exprs(self, ctx) -> typing.List[ast.Expr]:
        return [x for x in (child.accept(self) for child in ctx.getChildren()) if x]

    def visitParse_props(self, ctx):
        return self._parse_props(ctx.props)

    def visitParse_data(self, ctx):
        return [x for x in (child.accept(self) for child in ctx.getChildren()) if x]

    def visitFpcore(self, ctx) -> ast.FPCore:
        if ctx.ident is None:
            ident = None
        else:
            ident = ctx.ident.text

        inputs = [arg.accept(self) for arg in ctx.inputs]

        sanitize_arglist(inputs)

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
        return self._intern_rational(k)

    def visitExprSym(self, ctx) -> ast.Expr:
        return self._intern_symbol(ctx.x.text)

    def visitExprNum(self, ctx) -> ast.Expr:
        return ctx.n.accept(self)

    def visitExprAbort(self, ctx) -> ast.Expr:
        return ast.Abort()

    def visitExprDigits(self, ctx) -> ast.Expr:
        try:
            k = int(ctx.m.text), int(ctx.e.text), int(ctx.b.text)
        except ValueError:
            raise FPCoreParserError('digits: m, e, b must be integers, got {}, {}, {}'
                             .format(repr(ctx.m.text), repr(ctx.e.text), repr(ctx.b.text)))
        return self._intern_digits(k)

    def visitExprCtx(self, ctx) -> ast.Expr:
        return ast.Ctx(
            self._parse_props(ctx.props),
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
        return ctx.x.accept(self)

    def visitDatumStr(self, ctx) -> ast.Expr:
        return self._intern_string(ctx.s.text[1:-1])

    def visitDatumList(self, ctx) -> typing.Tuple[ast.Expr]:
        return tuple(d.accept(self) for d in ctx.data)

    def visitSymbolic(self, ctx) -> ast.Expr:
        return self._intern_symbol(ctx.x.text)


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

def parse_props(s):
    input_stream = antlr4.InputStream(s)
    lexer = FPCoreLexer(input_stream)
    token_stream = antlr4.CommonTokenStream(lexer)
    parser = FPCoreParser(token_stream)
    err_listener = LogErrorListener()
    parser.removeErrorListeners()
    parser.addErrorListener(err_listener)
    tree = parser.parse_props()
    errors = err_listener.syntax_errors
    if len(errors) > 0:
        err_text = ''.join('  ' + str(err) for err in errors)
        raise FPCoreParserError('unable to parse properties:\n' + err_text)
    else:
        return tree

def parse_data(s):
    input_stream = antlr4.InputStream(s)
    lexer = FPCoreLexer(input_stream)
    token_stream = antlr4.CommonTokenStream(lexer)
    parser = FPCoreParser(token_stream)
    err_listener = LogErrorListener()
    parser.removeErrorListeners()
    parser.addErrorListener(err_listener)
    tree = parser.parse_data()
    errors = err_listener.syntax_errors
    if len(errors) > 0:
        err_text = ''.join('  ' + str(err) for err in errors)
        raise FPCoreParserError('unable to parse data:\n' + err_text)
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
    if d.is_list():
        try:
            es = read_exprs(str(d))
        except FPCoreParserError:
            if strict:
                raise
            else:
                return None
        if len(es) == 1:
            return es[0]
        else:
            if strict:
                raise FPCoreParserError('data is not exactly one expression')
            else:
                return None
    elif d.is_string():
        if strict:
            raise FPCoreParserError('data is not exactly one expression')
        else:
            return None
    else:
        return d.value

def read_props(s):
    tree = parse_props(s)
    visitor = Visitor()
    return visitor.visit(tree)

def read_data(s):
    tree = parse_data(s)
    visitor = Visitor()
    return visitor.visit(tree)
