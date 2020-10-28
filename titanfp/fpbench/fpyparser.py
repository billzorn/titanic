import antlr4

from .fpcommon import *
from .FPYLexer import FPYLexer
from .FPYParser import FPYParser
from .FPYVisitor import FPYVisitor

class Visitor(FPYVisitor):

    def _parse_props(self, props):
        parsed = {}
        for prop in props:
            name, x = prop.accept(self)
            if self.check_names and name in parsed:
                raise FPCoreParserError(f'duplicate property {name!s}')
            parsed[name] = x
        return parsed

    def _create_ctx(self, e, props):
        if props:
            return ast.Ctx(props, e)
        else:
            return e

    def _create_var(self, e):
        # might want to change how this is handled at some point
        name = e.accept(self)
        if name in reserved_constants:
            return ast.Constant(name)
        else:
            return ast.Var(name)

    def _merge_bindings(self, *binding_groups, sequential=True):
        first_names = None
        bound_exprs = {}
        for bindings in binding_groups:
            names, es = zip(*bindings)
            if first_names is None:
                first_names = names
            else:
                if sequential:
                    if first_names != names:
                        raise FPCoreParserError(f'binding names do not agree, or are not in the same order')
                else: # not sequential
                    if set(first_names) != set(names):
                        raise FPCoreParserError(f'binding names do not agree')

            for name, e in bindings:
                if name in bound_exprs:
                    bound_exprs[name].append(e)
                else:
                    bound_exprs[name] = [e]

        return [(name, *bound_exprs[name]) for name in first_names]


    def __init__(self):
        super().__init__()
        self.check_names = True
        self.check_symbols = True


    def visitParse_fpy(self, ctx):
        return [x for x in (child.accept(self) for child in ctx.getChildren()) if x]

    def visitFpy(self, ctx):
        if ctx.ident is None:
            ident = None
        else:
            ident = ctx.ident.accept(self)

        args = ctx.args.accept(self)
        if self.check_names:
            sanitize_arglist(args)

        e, props = ctx.body.accept(self)

        name = props.get('name', None)
        # should also handle the other special metadata: pre and spec

        return ast.FPCore(args, e, props=props, ident=ident, name=name)

    def visitArglist(self, ctx):
        if ctx.arg is None:
            return []
        else:
            return [ctx.arg.accept(self), *(arg.accept(self) for arg in ctx.args)]

    def visitArgument(self, ctx):
        name = ctx.name.accept(self)
        if ctx.dims is None:
            shape = None
        else:
            shape = ctx.dims.accept(self)
        props = self._parse_props(ctx.props)

        return name, props, shape

    def visitDimlist(self, ctx):
        if ctx.dim is None:
            return []
        else:
            return [ctx.dim.accept(self), *(dim.accept(self) for dim in ctx.dims)]

    def visitDimension(self, ctx):
        if ctx.name is not None:
            return ctx.name.accept(self)
        else: # ctx.size is not None
            size = ctx.size.accept(self)
            if isinstance(size, ast.Integer):
                return size.i
            else:
                raise FPCoreParserError(f'fixed dimension {size!s} must be an integer')

    def visitNumber(self, ctx):
        if ctx.d is not None:
            k = ctx.d.text
            i = read_int(k)
            if i is not None:
                return ast.Integer(i)
            else:
                return ast.Decnum(k)
        elif ctx.x is not None:
            k = ctx.x.text
            i = read_int(k)
            if i is not None:
                return ast.Integer(i)
            else:
                return ast.Hexnum(k)
        else: # ctx.r is not None
            k = ctx.r.text
            p, q = ctx.r.text.split('/')
            return ast.Rational(p, q)

    def visitExpr(self, ctx):
        if ctx.e is not None:
            # expression is not a list
            return False, ctx.e.accept(self)
        else:
            # must be a list with some commas
            # head is the first element
            # rest may contain more elements (but is list either way)
            return True, [ctx.head.accept(self), *(e.accept(self) for e in ctx.rest)]

    def _one_expr(self, e):
        is_list, e = e.accept(self)
        if is_list:
            raise FPCoreParserError(f'expr cannot be a list here')
        return e

    def _some_exprs(self, e):
        if e is None:
            return []
        is_list, e = e.accept(self)
        if is_list:
            return e
        else:
            return [e]

    def visitNote(self, ctx):
        return self._create_ctx(ctx.e.accept(self), self._parse_props(ctx.props))

    def visitComp(self, ctx):
        if ctx.ops:
            # each group is of the form (opname, [e1, e2, ...])
            # where e1 etc. are the parse tree nodes, not ast expressions
            op_groups = []
            e_prev = ctx.e
            for e, op in zip(ctx.es, ctx.ops):
                opname = op.text
                if op_groups:
                    last_group = op_groups[-1]
                    if last_group[0] == opname:
                        last_group[1].append(e)
                    else:
                        op_groups.append((opname, [e_prev, e]))
                else:
                    op_groups.append((opname, [e_prev, e]))
                e_prev = e

            if len(op_groups) == 1:
                opname, es = op_groups[0]
                return reserved_constructs[opname](*(e.accept(self) for e in es))
            else:
                return ast.And(*(reserved_constructs[opname](*(e.accept(self) for e in es))
                                 for opname, es in op_groups))

        else:
            return ctx.e.accept(self)

    def _visit_binops(self, ctx):
        leftmost = ctx.e.accept(self)
        for e, op in zip(ctx.es, ctx.ops):
            opname = op.text
            leftmost = reserved_constructs[opname](leftmost, e.accept(self))
        return leftmost

    def visitArith(self, ctx):
        return self._visit_binops(ctx)

    def visitTerm(self, ctx):
        return self._visit_binops(ctx)

    def visitFactor(self, ctx):
        if ctx.op is not None:
            opname = ctx.op.text
            return reserved_constructs[opname](ctx.f.accept(self))
        else: # ctx.e is not None
            return ctx.e.accept(self)

    def visitPower(self, ctx):
        leftmost = ctx.e.accept(self)
        if ctx.op is not None:
            opname = op.text
            leftmost = reserved_constructs[opname](leftmost, ctx.f.accept(self))
        return leftmost

    def visitAtom(self, ctx):
        if ctx.x is not None:
            return self._create_var(ctx.x)
        elif ctx.n is not None:
            return ctx.n.accept(self)
        elif ctx.parens is not None:
            if ctx.e is None:
                raise FPCoreParserError(f'cannot have empty expression')
            return self._one_expr(ctx.e)
        elif ctx.bracks is not None:
            return ast.Array(*self._some_exprs(ctx.lst))
        elif ctx.call is not None:
            f = ctx.call.accept(self)

            if isinstance(f, ast.Var):
                name = f.value
            elif isinstance(f, ast.Constant):
                name = f.value
            else:
                raise FPCoreParserError(f'can only call a function name')

            # this is bad, because having the wrong number of arguments will crash without a parser error
            impl = reserved_constructs.get(name, None)
            if impl is not None:
                return impl(*self._some_exprs(ctx.args))
            else:
                return ast.UnknownOperator(*self._some_exprs(ctx.args), name=name)

        elif ctx.deref is not None:
            leftmost = ctx.deref.accept(self)
            return ast.Ref(leftmost, *self._some_exprs(ctx.args))
        elif ctx.dig is not None:
            args = self._some_exprs(ctx.digits)
            if len(args) != 3:
                raise FPCoreParserError(f'must supply 3 arguments to digits')
            meb = []
            for arg in args:
                if isinstance(arg, ast.Integer):
                    meb.append(arg.i)
                else:
                    raise FPCoreParserError(f'arguments to digits must be integers')
            return ast.Digits(*meb)
        else: # ctx.abort is not None
            return ast.Abort()

    def visitProp(self, ctx):
        return ctx.x.accept(self), ast.Data(ctx.d.accept(self))

    def visitSimple_stmt(self, ctx):
        return self._one_expr(ctx.e)

    def visitBinding(self, ctx):
        return ctx.x.accept(self), ctx.asgn.text, self._suite_ctx(ctx.body)

    def visitBlock(self, ctx):
        asgn_type = None
        bindings = []
        for name, asgn, e in (binding.accept(self) for binding in ctx.bindings):
            if asgn_type is None:
                asgn_type = asgn
            else:
                if asgn_type != asgn:
                    raise FPCoreParserError(f'all bindings in block must be of same type')
            bindings.append((name, e))
        sequential = (asgn_type == ':=')
        return sequential, bindings

    def visitIf_stmt(self, ctx):
        if ctx.test is not None:
            if_cond = self._one_expr(ctx.test)
            if_body = self._suite_ctx(ctx.body)
        else: # ctx.testsuite is not None
            if_cond = self._suite_ctx(ctx.testsuite)
            if_body = self._suite_ctx(ctx.bodysuite)

        cases = [(if_cond, if_body)]
        elif_tests = [*reversed(ctx.tests)]
        elif_bodies = [*reversed(ctx.bodies)]
        then_tests = [*reversed(ctx.testsuites)]
        then_bodies = [*reversed(ctx.bodysuites)]
        for tok in ctx.eliftypes:
            if tok.text == 'elif':
                elif_cond = self._one_expr(elif_tests.pop())
                elif_body = self._suite_ctx(elif_bodies.pop())
            else: # tok.text == 'then'
                elif_cond = self._suite_ctx(then_tests.pop())
                elif_body = self._suite_ctx(then_bodies.pop())
            cases.append((elif_cond, elif_body))

        rightmost = self._suite_ctx(ctx.else_body)
        while cases:
            cond, body = cases.pop()
            rightmost = ast.If(cond, body, rightmost)
        return rightmost

    def visitLet_stmt(self, ctx):
        sequential, bindings = ctx.bindings.accept(self)
        if sequential:
            return ast.LetStar(bindings, self._suite_ctx(ctx.body))
        else:
            return ast.Let(bindings, self._suite_ctx(ctx.body))

    def visitWhile_stmt(self, ctx):
        if ctx.test is not None:
            cond = self._one_expr(ctx.test)
        else: # ctx.testsuite is not None
            cond = self._suite_ctx(ctx.testsuite)
        iseq, inits = ctx.inits.accept(self)
        useq, updates = ctx.updates.accept(self)

        print(repr(cond))

        # this is direct transcription
        # we should do some analysis to allow syntactic sugar with missing variables etc.

        if not (iseq == useq):
            raise FPCoreParserError(f'while inits and updates must have same type of bindings')

        if iseq:
            return ast.WhileStar(cond,
                                 self._merge_bindings(inits, updates, sequential=iseq),
                                 self._suite_ctx(ctx.body))
        else:
            return ast.While(cond,
                             self._merge_bindings(inits, updates, sequential=iseq),
                             self._suite_ctx(ctx.body))

    def visitFor_stmt(self, ctx):
        dseq, dims = ctx.dims.accept(self) # note dseq is unused
        iseq, inits = ctx.inits.accept(self)
        useq, updates = ctx.updates.accept(self)
        if not (iseq == useq):
            raise FPCoreParserError(f'for inits and updates must have same type of bindings')

        if iseq:
            return ast.ForStar(dims,
                               self._merge_bindings(inits, updates, sequential=iseq),
                               self._suite_ctx(ctx.body))
        else:
            return ast.For(dims,
                           self._merge_bindings(inits, updates, sequential=iseq),
                           self._suite_ctx(ctx.body))

    def visitTensor_stmt(self, ctx):
        dseq, dims = ctx.dims.accept(self)
        if ctx.inits is not None:
            if not dseq:
                raise FPCoreParserError(f'can only have loop updates on a tensor* with sequential bindings')
            iseq, inits = ctx.inits.accept(self)
            useq, updates = ctx.updates.accept(self)
            if not (iseq and useq):
                raise FPCoreParserError(f'all bindings must be sequential for tensor*')
        else:
            inits = []
            updates = []

        if dseq:
            return ast.TensorStar(dims,
                                  self._merge_bindings(inits, updates, sequential=dseq),
                                  self._suite_ctx(ctx.body))
        else:
            return ast.Tensor(dims,
                              self._suite_ctx(ctx.body))

    def _visit_one_child(self, ctx):
        e, = [x for x in (child.accept(self) for child in ctx.getChildren()) if x]
        return e

    def visitCompound_stmt(self, ctx):
        return self._visit_one_child(ctx)

    def visitStatement(self, ctx):
        return self._visit_one_child(ctx)

    def visitDatum(self, ctx):
        if ctx.x is not None:
            return self._create_var(ctx.x)
        elif ctx.n is not None:
            return ctx.n.accept(self)
        elif ctx.s is not None:
            return ast.String(ctx.s.text[1:-1])
        else: # ctx.data is not None
            return tuple(d.accept(self) for d in ctx.data)

    def visitSimple_data(self, ctx):
        if len(ctx.data) == 1:
            return ctx.data[0].accept(self)
        else:
            return tuple(d.accept(self) for d in ctx.data)

    def visitData_suite(self, ctx):
        if ctx.data is not None:
            return ast.Data(ctx.data.accept(self))
        else: # ctx.body is not None
            return ast.Data(self._suite_ctx(ctx.body))

    def visitAnnotation(self, ctx):
        return ctx.x.accept(self), ctx.data.accept(self)

    def visitSuite(self, ctx):
        if ctx.e is not None:
            return ctx.e.accept(self), {}
        else: # ctx.body is not None
            props = self._parse_props(ctx.props)
            return ctx.body.accept(self), props

    def _suite_ctx(self, body):
        e, props = body.accept(self)
        return self._create_ctx(e, props)

    def _visit_sym(self, ctx):
        if ctx.x is not None:
            return ctx.x.text
        else: # ctx.s_str is not None
            sym_text = ctx.s_str.text[2:-1]
            if self.check_symbols:
                sanitize_symbol(sym_text)
            return sym_text

    def visitSymbolic(self, ctx):
        return self._visit_sym(ctx)

    def visitSymbolic_data(self, ctx):
        return self._visit_sym(ctx)



class LogErrorListener(antlr4.error.ErrorListener.ErrorListener):

    def __init__(self):
        super().__init__()
        self.syntax_errors = []

    def syntaxError(self, recognizer, offending_symbol, line, column, msg, e):
        self.syntax_errors.append("line " + str(line) + ":" + str(column) + " " + msg)


def dump_token_stream(token_stream):
    batch = 10000
    fetched = token_stream.fetch(batch)
    total = fetched
    while fetched > 0:
        fetched = token_stream.fetch(batch)
        total += fetched
    print(f'fetched {total!s} tokens in total.')

    for i, tok in enumerate(token_stream.tokens):
        print(i, tok)

    print(flush=True)


def parse(s):
    input_stream = antlr4.InputStream(s)
    lexer = FPYLexer(input_stream)
    token_stream = antlr4.CommonTokenStream(lexer)

    dump_token_stream(token_stream)

    parser = FPYParser(token_stream)
    err_listener = LogErrorListener()
    parser.removeErrorListeners()
    parser.addErrorListener(err_listener)
    tree = parser.parse_fpy()
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






example = """FPCore foo(x, y, z[2,2]):
  with:
    a = 0
    b = 1
  while a < x:
    a = a + 1
    b = b + b
  in:
    b
"""

example = """FPCore (x, y):
  fma(x, y, x)
"""
