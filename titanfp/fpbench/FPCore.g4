grammar FPCore;

parse : fpcore* EOF ;

// We break the standard by allowing cores to cores to have names,
// which can be referred to as operators in other cores.
fpcore : '(' 'FPCore' (cid=SYMBOL)? '(' (inputs+=SYMBOL)* ')' (props+=prop)* e=expr ')' ;

expr
    : n=NUMBER # ExprNumeric
        // A symbolic expression can be either a variable or a constant.
    | x=SYMBOL # ExprSymbolic
        // if statements look like applications, so we give them
        // higher precedence.
    | '(' 'if' cond=expr then_body=expr else_body=expr ')'                                 # ExprIf
    | '(' 'let' '(' ('[' xs+=SYMBOL es+=expr ']')* ')' body=expr ')'                       # ExprLet
    | '(' 'while' cond=expr '(' ('[' xs+=SYMBOL e0s+=expr es+=expr ']')* ')' body=expr ')' # ExprWhile
        // We allow nullary application here, as function calls to other cores are
        // the same as primitive operators according to this grammar.
    | '(' op=SYMBOL (args+=expr)* ')' # ExprOp
    ;

prop
    : name=SYMBOL s=STRING              # PropStr
    | name=SYMBOL '(' (xs+=SYMBOL)* ')' # PropList
    | name=SYMBOL e=expr                # PropExpr
    ;

// As described in the FPCore 1.0 standard, with some minor fixes.
NUMBER : [-+]? [0-9]+ ('.' [0-9]+)? ('e' [-+]?[0-9]+)? ;
SYMBOL : [a-zA-Z~!@$%^&*_\-+=<>.?/:] [a-zA-Z0-9~!@$%^&*_\-+=<>.?/:]* ;
STRING : '"' ([\u0020-\u0021\u0023-\u005b\u005d-\u007e] | '\\' [\u0022\u005c])* '"' ;

WS : [ \t\n\r]+ -> skip ;
UNK : . ;
