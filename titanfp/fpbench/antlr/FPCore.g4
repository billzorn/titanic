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

fpcore : OPEN FPCORE (ident=SYMBOL)? OPEN (inputs+=argument)* CLOSE (props+=prop)* e=expr CLOSE ;

dimension
    : name=SYMBOL # DimSym
    | size=number # DimSize
    ;

argument
    : name=SYMBOL
    | OPEN name=SYMBOL (shape+=dimension)+ CLOSE
    | OPEN ANNOTATION (props+=prop)* name=SYMBOL (shape+=dimension)* CLOSE
    ;

number
    : n=DECNUM # NumberDec
    | n=HEXNUM # NumberHex
    | n=RATIONAL # NumberRational
    ;

expr
    : x=SYMBOL # ExprSym
    | n=number # ExprNum
    | ABORT # ExprAbort
    | OPEN DIGITS m=DECNUM e=DECNUM b=DECNUM CLOSE # ExprDigits
    | OPEN ANNOTATION (props+=prop)* body=expr CLOSE # ExprCtx
    | OPEN IF cond=expr then_body=expr else_body=expr CLOSE # ExprIf
    | OPEN LET OPEN (OPEN xs+=SYMBOL es+=expr CLOSE)* CLOSE body=expr CLOSE # ExprLet
    | OPEN LETSTAR OPEN (OPEN xs+=SYMBOL es+=expr CLOSE)* CLOSE body=expr CLOSE # ExprLetStar
    | OPEN WHILE cond=expr OPEN (OPEN xs+=SYMBOL e0s+=expr es+=expr CLOSE)* CLOSE body=expr CLOSE # ExprWhile
    | OPEN WHILESTAR cond=expr OPEN (OPEN xs+=SYMBOL e0s+=expr es+=expr CLOSE)* CLOSE body=expr CLOSE # ExprWhileStar
    | OPEN FOR OPEN (OPEN xs+=SYMBOL es+=expr CLOSE)* CLOSE
                 OPEN (OPEN while_xs+=SYMBOL while_e0s+=expr while_es+=expr CLOSE)* CLOSE body=expr CLOSE # ExprFor
    | OPEN FORSTAR OPEN (OPEN xs+=SYMBOL es+=expr CLOSE)* CLOSE
                  OPEN (OPEN while_xs+=SYMBOL while_e0s+=expr while_es+=expr CLOSE)* CLOSE body=expr CLOSE # ExprForStar
    | OPEN TENSOR OPEN (OPEN xs+=SYMBOL es+=expr CLOSE)* CLOSE body=expr CLOSE # ExprTensor
    | OPEN TENSORSTAR (name=SYMBOL)? OPEN (OPEN xs+=SYMBOL es+=expr CLOSE)* CLOSE
                                   (OPEN (OPEN while_xs+=SYMBOL while_e0s+=expr while_es+=expr CLOSE)* CLOSE)?
                                    body=expr CLOSE # ExprTensorStar
    | OPEN SUGAR_INT body=expr CLOSE # ExprSugarInt
    | OPEN op=SYMBOL (args+=expr)* CLOSE # ExprOp
    ;

// Keywords in properties are not required by the grammar to start with a colon;
// it's up to the visitor to check that.
prop : name=SYMBOL d=datum ;

datum
    : x=symbolic # DatumSym
    | n=number # DatumNum
    | s=STRING # DatumStr
    | OPEN (data+=datum)* CLOSE # DatumList
    ;

// Reserved symbols (like FPCore or if) tokenize differently from "normal" SYMBOLs;
// in some cases (like data) we want to permit both.
symbolic
    : x=FPCORE
    | x=ABORT
    | x=DIGITS
    | x=ANNOTATION
    | x=IF
    | x=LET
    | x=LETSTAR
    | x=WHILE
    | x=WHILESTAR
    | x=FOR
    | x=FORSTAR
    | x=TENSOR
    | x=TENSORSTAR
    | x=SUGAR_INT
    | x=SYMBOL
    ;


// Tokens

OPEN  : '(' | '[' ;
CLOSE : ')' | ']' ;

// All of these constructs need to be declared explicitly, to control the precedence
// in lexer rules.
FPCORE     : 'FPCore' ;
ABORT      : 'abort' ;
DIGITS     : 'digits' ;
ANNOTATION : '!' ;
IF         : 'if' ;
LET        : 'let' ;
LETSTAR    : 'let*' ;
WHILE      : 'while' ;
WHILESTAR  : 'while*' ;
FOR        : 'for' ;
FORSTAR    : 'for*' ;
TENSOR     : 'tensor' ;
TENSORSTAR : 'tensor*' ;
SUGAR_INT  : '#' ;

DECNUM   : [+-]? ([0-9]+ ('.' [0-9]+)? | '.' [0-9]+) ([eE] [-+]? [0-9]+)? ;
HEXNUM   : [+-]? '0' [xX] ([0-9a-fA-F]+ ('.' [0-9a-fA-F]+)? | '.' [0-9a-fA-F]+) ([pP] [-+]? [0-9]+)? ;
RATIONAL : [+-]? [0-9]+ '/' [0-9]* [1-9] [0-9]* ;

SYMBOL : [a-zA-Z~!@$%^&*_\-+=<>.?/:] [a-zA-Z0-9~!@$%^&*_\-+=<>.?/:]* ;
STRING : '"' ([\u0008-\u000d\u0020-\u0021\u0023-\u005b\u005d-\u007e] | '\\' [bfnrtv\u0022\u005c])* '"' ;

WS : [ \t\n\r]+ -> skip ;
// Racket allows block comments to be nested; that may not work here.
BLOCK_COMMENT : '#|' .*? '|#' -> skip ;
LINE_COMMENT : ';' ~[\r\n]* -> skip ;
UNK : . ;
