grammar FPCore;

parse : fpcore* EOF ;

fpcore : '(' 'FPCore' '(' (inputs+=argument)* ')' (props+=prop)* e=expr ')' ;

argument
    : name=SYMBOL
    | ( '!' (props+=prop)* name=SYMBOL )
    ;

rounded_expr
    : n=NUMBER # RoundedNumber
    | n=HEXNUM # RoundedHexnum
    // Variables and constants both appear as symbols.
    | x=SYMBOL # RoundedSymbolic
    | '(' 'digits' m=NUMBER e=NUMBER b=NUMBER ')' # RoundedDigits
    | '(' op=SYMBOL (args+=expr)* ')' # RoundedOp
    ;

expr
    : '(' 'if' cond=expr then_body=expr else_body=expr ')' # ExprIf
    | '(' 'let' '(' ('[' xs+=SYMBOL es+=expr ']')* ')' body=expr ')' # ExprLet
    | '(' 'while' cond=expr '(' ('[' xs+=SYMBOL e0s+=expr es+=expr ']')* ')' body=expr ')' # ExprWhile
    | '(' '!' (props+=prop)* body=rounded_expr ')' # ExprExplicit
    // This form should go last, as if statements can be confused with operations.
    | body=rounded_expr # ExprImplicit
    ;

// Keywords in properties are not required by the grammar to start with a colon;
// it's up to the visitor to check that.
prop
    : name=SYMBOL s=STRING # PropStr
    | name=SYMBOL '(' (xs+=SYMBOL)* ')' # PropList
    | name=SYMBOL e=expr # PropExpr
    ;

NUMBER : [-+]? [0-9]+ ('.' [0-9]+)? ('e' [-+]?[0-9]+)? ;
HEXNUM : [-+]? '0' [xX] [0-9a-fA-F]+ ('.' [0-9a-fA-F]+)? ('p' [-+]?[0-9]+) ;
SYMBOL : [a-zA-Z~!@$%^&*_\-+=<>.?/:] [a-zA-Z0-9~!@$%^&*_\-+=<>.?/:]* ;
STRING : '"' ([\u0020-\u0021\u0023-\u005b\u005d-\u007e] | '\\' [\u0022\u005c])* '"' ;

WS : [ \t\n\r]+ -> skip ;
UNK : . ;
