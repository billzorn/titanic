grammar FPY;

// see: https://github.com/antlr/grammars-v4/blob/master/python/python3-py/Python3.g4

tokens { INDENT, DEDENT }

@lexer::header{
from antlr4.Token import CommonToken
import re
import importlib

# Pre-compile re
only_spaces = re.compile("[^\r\n\f]+")
only_newlines = re.compile("[\r\n\f]+")

# Allow languages to extend the lexer and parser, by loading the parser dynamically
module_path = __name__[:-5]
language_name = __name__.split('.')[-1]
language_name = language_name[:-5]  # Remove Lexer from name
LanguageParser = getattr(importlib.import_module('{}Parser'.format(module_path)), '{}Parser'.format(language_name))
}

@lexer::members {
@property
def tokens(self):
    try:
        return self._tokens
    except AttributeError:
        self._tokens = []
        return self._tokens

@property
def indents(self):
    try:
        return self._indents
    except AttributeError:
        self._indents = []
        return self._indents

@property
def opened(self):
    try:
        return self._opened
    except AttributeError:
        self._opened = 0
        return self._opened

@opened.setter
def opened(self, value):
    self._opened = value

@property
def lastToken(self):
    try:
        return self._lastToken
    except AttributeError:
        self._lastToken = None
        return self._lastToken

@lastToken.setter
def lastToken(self, value):
    self._lastToken = value

def reset(self):
    super().reset()
    self.tokens = []
    self.indents = []
    self.opened = 0
    self.lastToken = None

def emitToken(self, t):
    super().emitToken(t)
    self.tokens.append(t)

def nextToken(self):
    if self._input.LA(1) == Token.EOF and self.indents:
        for i in range(len(self.tokens)-1,-1,-1):
            if self.tokens[i].type == Token.EOF:
                self.tokens.pop(i)

        self.emitToken(self.commonToken(LanguageParser.NEWLINE, '\n'))
        while self.indents:
            self.emitToken(self.createDedent())
            self.indents.pop()

        self.emitToken(self.commonToken(LanguageParser.EOF, "<EOF>"))
    next = super().nextToken()
    if next.channel == Token.DEFAULT_CHANNEL:
        self.lastToken = next
    return next if not self.tokens else self.tokens.pop(0)

def createDedent(self):
    dedent = self.commonToken(LanguageParser.DEDENT, "")
    dedent.line = self.lastToken.line
    return dedent

def commonToken(self, type, text, indent=0):
    stop = self.getCharIndex()-1-indent
    start = (stop - len(text) + 1) if text else stop
    return CommonToken(self._tokenFactorySourcePair, type, super().DEFAULT_TOKEN_CHANNEL, start, stop)

@staticmethod
def getIndentationCount(spaces):
    count = 0
    for ch in spaces:
        if ch == '\t':
            count += 8 - (count % 8)
        else:
            count += 1
    return count

def atStartOfInput(self):
    return Lexer.column.fget(self) == 0 and Lexer.line.fget(self) == 1
}


// entry rules

parse_fpy : fpy* EOF ;

// grammar

fpy : FPCORE (ident=symbolic)? args=arglist COLON body=suite;

arglist : OPEN_PAREN (arg=argument (COMMA args+=argument)* COMMA?)? CLOSE_PAREN;

argument : name=symbolic dims=dimlist? (BANG props+=prop)*;

dimlist : OPEN_BRACK (dim=dimension (COMMA dims+=dimension)* COMMA?)? CLOSE_BRACK;

dimension
    : name=symbolic
    | size=number
    ;

number
    : d=DECNUM
    | x=HEXNUM
    | r=RATIONAL
    ;

expr
    : e=note
    | head=expr COMMA
    | head=expr (COMMA rest+=expr)+ COMMA?
    ;

note   : e=comp (BANG props+=prop)* ;
comp   : e=arith (ops+=(LT | LE | GT | GE | EQ | NE) es+=arith)*;
arith  : e=term (ops+=(PLUS | MINUS) es+=term)* ;
term   : e=factor (ops+=(TIMES | DIVIDE | MOD) es+=factor)* ;
factor : op=(PLUS | MINUS) f=factor | e=power ;
power  : e=atom (POWER f=factor)? ;

atom 
    : x=symbolic
    | n=number
    | OPEN_PAREN (e=expr)? CLOSE_PAREN
    | OPEN_BRACK (lst=expr?) CLOSE_BRACK
    | call=atom OPEN_PAREN (args=expr)? CLOSE_PAREN
    | deref=atom OPEN_BRACK (args=expr)? CLOSE_BRACK
    | DIG OPEN_PAREN (digits=expr)? CLOSE_PAREN
    | ABORT
    ;

prop : x=symbolic d=datum ;

simple_stmt : expr NEWLINE ;

binding
    : x=symbolic asgn=IS   body=suite
    | x=symbolic asgn=GETS body=suite
    ;

block : NEWLINE INDENT binding+ DEDENT ;

if_stmt : (((IF test=expr COLON body=suite)
           |(IF COLON testsuite=suite THEN COLON bodysuite=suite))
           ((ELIF tests+=expr COLON bodies+=suite)
           |(ELIF COLON testsuites+=suite THEN COLON bodysuites+=suite))*
           ELSE COLON else_body=suite)
        ;

let_stmt : (LET COLON bindings=block
            IN  COLON body=suite)
         ;

while_stmt : (WITH              COLON inits=block
              ((WHILE test=expr COLON updates=block)
              |(WHILE           COLON testsuite=suite 
                DO              COLON updates=block))
              IN                COLON body=suite)
           ;

for_stmt : (FOR  COLON dims=block
            WITH COLON inits=block
            DO   COLON updates=block
            IN   COLON body=suite)
         ;

tensor_stmt : (TENSOR COLON dims=block
               (WITH  COLON inits=block
                DO    COLON updates=block)?
               OF     COLON body=suite)
            ;

compound_stmt
    : if_stmt
    | let_stmt
    | while_stmt
    | for_stmt
    | tensor_stmt
    ;

statement
    : simple_stmt 
    | compound_stmt
    ;

datum
    : x=symbolic_data
    | n=number
    | s=STRING
    | open_ (data+=datum)* close_
    ;

simple_data : (data+=datum)+ NEWLINE ;

data_suite
    : data=simple_data
    | body=suite
    ;

annotation : x=symbolic COLON data=data_suite ;

data_block : NEWLINE INDENT (props+=annotation)+ DEDENT ;

suite
    : e=simple_stmt
    | NEWLINE INDENT (props+=annotation)* body=statement DEDENT
    ;

symbolic : SYMBOL | SYM OPEN_PAREN (s=STRING) CLOSE_PAREN ;

symbolic_data
    : x=FPCORE
    | x=IF
    | x=THEN
    | x=ELIF
    | x=ELSE
    | x=LET
    | x=WHILE
    | x=FOR
    | x=TENSOR
    | x=WITH
    | x=DO
    | x=IN
    | x=OF
    | x=IS
    | x=GETS
    | x=COLON
    | x=COMMA
    | x=BANG
    | x=POWER
    | x=PLUS
    | x=MINUS
    | x=TIMES
    | x=DIVIDE
    | x=MOD
    | x=LT
    | x=LE
    | x=GT
    | x=GE
    | x=EQ
    | x=NE
    | x=ABORT
    | x=SYM
    | x=DIG
    | x=SYMBOL
    | x=S_SYMBOL
    ;

open_  : OPEN_PAREN | OPEN_BRACK ;
close_ : CLOSE_PAREN | CLOSE_BRACK ;


// tokens

OPEN_PAREN : '(' {self.opened += 1};
CLOSE_PAREN : ')' {self.opened -= 1};
OPEN_BRACK : '[' {self.opened += 1};
CLOSE_BRACK : ']' {self.opened -= 1};

// All of these constructs need to be declared explicitly, to control the precedence
// in lexer rules.
FPCORE     : 'FPCore' ;
IF         : 'if' ;
THEN       : 'then' ;
ELIF       : 'elif' ;
ELSE       : 'else' ;
LET        : 'let' ;
WHILE      : 'while' ;
FOR        : 'for' ;
TENSOR     : 'tensor' ;
WITH       : 'with' ;
DO         : 'do' ;
IN         : 'in' ;
OF         : 'of' ;

IS         : '=' ;
GETS       : ':=' ;
COLON      : ':' ;
COMMA      : ',' ;
BANG       : '!' ;

POWER      : '**' ;
PLUS       : '+' ;
MINUS      : '-' ;
TIMES      : '*' ;
DIVIDE     : '/' ;
MOD        : '%' ;
LT         : '<' ;
LE         : '<=' ;
GT         : '>' ;
GE         : '>=' ;
EQ         : '==' ;
NE         : '!=' ;

ABORT      : 'abort' ;
SYM        : 'symbol' ;
DIG        : 'digits' ;

DECNUM   : [+-]? ([0-9]+ ('.' [0-9]+)? | '.' [0-9]+) ([eE] [-+]? [0-9]+)? ;
HEXNUM   : [+-]? '0' [xX] ([0-9a-fA-F]+ ('.' [0-9a-fA-F]+)? | '.' [0-9a-fA-F]+) ([pP] [-+]? [0-9]+)? ;
RATIONAL : [+-]? [0-9]+ '/' [0-9]* [1-9] [0-9]* ;

// For the infix grammar, we want to trim this down a bit. Must go first for precedence.
SYMBOL : [a-zA-Z~@$^&_.?] [a-zA-Z0-9~@$^&_.?]* ;

// From the original S-expression grammar.
S_SYMBOL : [a-zA-Z~!@$%^&*_\-+=<>.?/:] [a-zA-Z0-9~!@$%^&*_\-+=<>.?/:]* ;

STRING : '"' ([\u0008-\u000d\u0020-\u0021\u0023-\u005b\u005d-\u007e] | '\\' [bfnrtv\u0022\u005c])* '"' ;

NEWLINE
 : ( {self.atStartOfInput()}?   SPACES
   | ( '\r'? '\n' | '\r' | '\f' ) SPACES?
   )
   {
tempt = Lexer.text.fget(self)
newLine = only_spaces.sub("", tempt)
spaces = only_newlines.sub("", tempt)
la_char = ""
try:
    la = self._input.LA(1)
    la_char = chr(la)
except ValueError: # End of file
    pass

# Strip newlines inside open clauses except if we are near EOF. We keep NEWLINEs near EOF to
# satisfy the final newline needed by the single_put rule used by the REPL.
try:
    nextnext_la = self._input.LA(2)
    nextnext_la_char = chr(nextnext_la)
except ValueError:
    nextnext_eof = True
else:
    nextnext_eof = False

if self.opened > 0 or nextnext_eof is False and (la_char == '\r' or la_char == '\n' or la_char == '\f' or la_char == '#'):
    self.skip()
else:
    indent = self.getIndentationCount(spaces)
    previous = self.indents[-1] if self.indents else 0
    self.emitToken(self.commonToken(self.NEWLINE, newLine, indent=indent))      # NEWLINE is actually the '\n' char
    if indent == previous:
        self.skip()
    elif indent > previous:
        self.indents.append(indent)
        self.emitToken(self.commonToken(LanguageParser.INDENT, spaces))
    else:
        while self.indents and self.indents[-1] > indent:
            self.emitToken(self.createDedent())
            self.indents.pop()
    }
 ;

SKIP_
 : ( SPACES | COMMENT | LINE_JOINING ) -> skip
 ;

UNK_
 : .
 ;


// fragments

fragment SPACES
 : [ \t]+
 ;

fragment COMMENT
 : '#' ~[\r\n\f]*
 ;

fragment LINE_JOINING
 : '\\' SPACES? ( '\r'? '\n' | '\r' | '\f' )
 ;