import antlr4

from .fpcommon import *
from .FPYLexer import FPYLexer
from .FPYParser import FPYParser
from .FPYVisitor import FPYVisitor

class Visitor(FPYVisitor):

    def visitParse_fpy(self, ctx):
        return [x for x in (child.accept(self) for child in ctx.getChildren()) if x]

    def visitFpy(self, ctx):
        if ctx.ident is None:
            ident = None
        else:
            ident = ctx.ident.text

        args = ctx.args.accept(self)

        


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

        
def parse(s):
    input_stream = antlr4.InputStream(s)
    lexer = FPYLexer(input_stream)
    token_stream = antlr4.CommonTokenStream(lexer)

    dump_token_stream(token_stream)

    parser = FPYParser(token_stream)
    err_listener = LogErrorListener()
    parser.removeErrorListeners()
    parser.addErrorListener(err_listener)
    tree = parser.file_input()
    errors = err_listener.syntax_errors
    if len(errors) > 0:
        err_text = ''.join('  ' + str(err) for err in errors)
        raise ValueError('unable to parse FPCore:\n' + err_text)
    else:
        return tree

def go(s):
    tree = parse(s)
    visitor = Visitor()
    return visitor.visit(tree)






example = """x = 3
while x < 10:
  x += 1
  foo = 0
x"""
