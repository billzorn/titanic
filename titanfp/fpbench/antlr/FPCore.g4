grammar FPCore;

// main entrypoint for .fpcore benchmark files
parse_fpcore : fpcore* EOF ;

// secondary entrypoint for parsing arguments, preconditions, etc.
parse_exprs : expr* EOF ;

// secondary entrypoint for parsing lists of properties
parse_props : (props+=prop)* EOF ;

// secondary entrypoint for parsing data
parse_data : datum* EOF ;


// FPCore grammar, implementing the standard from fpbench.org

fpcore : OPEN 'FPCore' (ident=SYMBOL)? OPEN (inputs+=argument)* CLOSE (props+=prop)* e=expr CLOSE ;

dimension
    : name=SYMBOL # DimSym
    | size=number # DimSize
    ;

argument
    : name=SYMBOL
    | OPEN name=SYMBOL (shape+=dimension)+ CLOSE
    | OPEN '!' (props+=prop)* name=SYMBOL (shape+=dimension)* CLOSE
    ;

expr
    : n=number # ExprNum
    | x=SYMBOL # ExprSym
    | OPEN '!' (props+=prop)* body=expr CLOSE # ExprCtx
    | OPEN 'tensor' OPEN (OPEN xs+=SYMBOL es+=expr CLOSE)* CLOSE body=expr CLOSE # ExprTensor
    | OPEN 'tensor*' (name=SYMBOL)? OPEN (OPEN xs+=SYMBOL es+=expr CLOSE)* CLOSE
                                   (OPEN (OPEN while_xs+=SYMBOL while_e0s+=expr while_es+=expr CLOSE)* CLOSE)?
                                    body=expr CLOSE # ExprTensorStar
    | OPEN 'if' cond=expr then_body=expr else_body=expr CLOSE # ExprIf
    | OPEN 'let' OPEN (OPEN xs+=SYMBOL es+=expr CLOSE)* CLOSE body=expr CLOSE # ExprLet
    | OPEN 'let*' OPEN (OPEN xs+=SYMBOL es+=expr CLOSE)* CLOSE body=expr CLOSE # ExprLetStar
    | OPEN 'while' cond=expr OPEN (OPEN xs+=SYMBOL e0s+=expr es+=expr CLOSE)* CLOSE body=expr CLOSE # ExprWhile
    | OPEN 'while*' cond=expr OPEN (OPEN xs+=SYMBOL e0s+=expr es+=expr CLOSE)* CLOSE body=expr CLOSE # ExprWhileStar
    | OPEN 'for' OPEN (OPEN xs+=SYMBOL es+=expr CLOSE)* CLOSE
                 OPEN (OPEN while_xs+=SYMBOL while_e0s+=expr while_es+=expr CLOSE)* CLOSE body=expr CLOSE # ExprFor
    | OPEN 'for*' OPEN (OPEN xs+=SYMBOL es+=expr CLOSE)* CLOSE
                  OPEN (OPEN while_xs+=SYMBOL while_e0s+=expr while_es+=expr CLOSE)* CLOSE body=expr CLOSE # ExprForStar
    | OPEN 'data' d=datum CLOSE # ExprData
    | OPEN '#' body=expr CLOSE # ExprSugarInt
    | OPEN op=SYMBOL (args+=expr)* CLOSE # ExprOp
    ;

number
    : n=DECNUM # NumberDec
    | n=HEXNUM # NumberHex
    | n=RATIONAL # NumberRational
    | OPEN 'digits' m=DECNUM e=DECNUM b=DECNUM CLOSE # NumberDigits
    ;

// Keywords in properties are not required by the grammar to start with a colon;
// it's up to the visitor to check that.
prop : name=SYMBOL d=datum ;

datum
    // : n=number # DatumNum
    // | x=SYMBOL # DatumSym
    : e=expr # DatumExpr
    | s=STRING # DatumStr
    | OPEN (data+=datum)* CLOSE # DatumList
    ;

// Some tokens.

OPEN : '(' | '[' ;
CLOSE : ')' | ']' ;

DECNUM : [+-]? ([0-9]+ ('.' [0-9]+)? | '.' [0-9]+) ([eE] [-+]? [0-9]+)? ;
HEXNUM : [+-]? '0' [xX] ([0-9a-fA-F]+ ('.' [0-9a-fA-F]+)? | '.' [0-9a-fA-F]+) ([pP] [-+]? [0-9]+)? ;
RATIONAL : [+-]? [0-9]+ '/' [0-9]* [1-9] [0-9]* ;

SYMBOL : [a-zA-Z~!@$%^&*_\-+=<>.?/:] [a-zA-Z0-9~!@$%^&*_\-+=<>.?/:]* ;
STRING : '"' ([\u0008-\u000d\u0020-\u0021\u0023-\u005b\u005d-\u007e] | '\\' [bfnrtv\u0022\u005c])* '"' ;

WS : [ \t\n\r]+ -> skip ;
// Racket allows block comments to be nested; that may not work here.
BLOCK_COMMENT : '#|' .*? '|#' -> skip ;
LINE_COMMENT : ';' ~[\r\n]* -> skip ;
UNK : . ;
