grammar FPCore;

parse : fpcore* EOF ;

fpcore : OPEN 'FPCore' OPEN (inputs+=argument)* CLOSE (props+=prop)* e=expr CLOSE ;

argument
    : name=SYMBOL
    | OPEN '!' (props+=prop)* name=SYMBOL CLOSE
    ;

expr
    : n=number #ExprNum
    | x=SYMBOL #ExprSym
    | OPEN '!' (props+=prop)* body=expr CLOSE #ExprCtx
    | OPEN 'cast' body=expr CLOSE #ExprCast
    | OPEN 'if' cond=expr then_body=expr else_body=expr CLOSE # ExprIf
    | OPEN 'let' OPEN (OPEN xs+=SYMBOL es+=expr CLOSE)* CLOSE body=expr CLOSE # ExprLet
    | OPEN 'while' cond=expr OPEN (OPEN xs+=SYMBOL e0s+=expr es+=expr CLOSE)* CLOSE body=expr CLOSE # ExprWhile
    | OPEN op=SYMBOL (args+=expr)* CLOSE # ExprOp
    ;

number
    : n=DECNUM
    | n=HEXNUM
    | n=RATIONAL
    | OPEN 'digits' m=DECNUM e=DECNUM b=DECNUM CLOSE
    ;

// Keywords in properties are not required by the grammar to start with a colon;
// it's up to the visitor to check that.
prop
    : name=SYMBOL s=STRING # PropStr
    | name=SYMBOL OPEN (xs+=SYMBOL)* CLOSE # PropList
    // Properties specified as symbols will end up being parsed as symbolic expressions.
    | name=SYMBOL e=expr # PropExpr
    ;

OPEN : '(' | '[' ;
CLOSE : ')' | ']' ;

DECNUM : [-+]? ([0-9]+ ('.' [0-9]+)? | '.' [0-9]+) ([eE] [-+]? [0-9]+)? ;
HEXNUM : [+-]? '0' [xX] ([0-9a-fA-F]+ ('.' [0-9a-fA-F]+)? | '.' [0-9a-fA-F]+) ([pP] [-+]? [0-9]+)? ;
RATIONAL : [+-]? [0-9]+ '/' [0-9]* [1-9] [0-9]* ;

SYMBOL : [a-zA-Z~!@$%^&*_\-+=<>.?/:] [a-zA-Z0-9~!@$%^&*_\-+=<>.?/:]* ;
STRING : '"' ([\u0020-\u0021\u0023-\u005b\u005d-\u007e] | '\\' [\u0022\u005c])* '"' ;

WS : [ \t\n\r]+ -> skip ;
// Racket allows block comments to be nested; that may not work here.
BLOCK_COMMENT : '#|' .*? '|#' -> skip ;
LINE_COMMENT : ';' ~[\r\n]* -> skip ;
UNK : . ;
