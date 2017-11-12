grammar FPCore;

parse
    : fpcore* EOF
    | fpimp* EOF
    ;

fpcore : '(' 'FPCore' '(' (inputs+=SYMBOL)* ')' (props+=prop)* e=expr ')' ;

// fpimp : '(' 'FPImp' ('(' SYMBOL ')')* prop* cmd* '(' 'output' expr* ')' ')' ;
fpimp : '(' 'FPImp' ')' ;

expr
    : c=NUMBER   # ExprNum
    | c=constant # ExprConst
    | x=SYMBOL   # ExprVar
    | '(' op=unop arg0=expr ')'                                     # ExprUnop
    | '(' op=binop arg0=expr arg1=expr ')'                          # ExprBinop
    | '(' op=comp args+=expr (args+=expr)+ ')'                      # ExprComp
    | '(' op=logical (args+=expr)+ ')'                              # ExprLogical
    | '(' 'if' expr expr expr ')'                                   # ExprIf
    | '(' 'let' '(' ('[' SYMBOL expr ']')* ')' expr ')'             # ExprLet
    | '(' 'while' expr '(' ('[' SYMBOL expr expr ']')* ')' expr ')' # ExprWhile
    ;

prop
    : name=SYMBOL s=STRING                # PropStr
    | name=SYMBOL '(' (syms+=SYMBOL)* ')' # PropList
    | name=SYMBOL e=expr                  # PropExpr
    ;

unop
    : '-'
    | 'sqrt'
    | 'not'
    ;

binop
    : '+'
    | '-'
    | '*'
    | '/'
    ;

comp
    : '<'
    | '>'
    | '<='
    | '>='
    | '=='
    | '!='
    ;

logical
    : 'and'
    | 'or'
    ;

constant
    : 'E'
    | 'LOG2E'
    | 'LOG10E'
    | 'LN2'
    | 'LN10'
    | 'PI'
    | 'PI_2'
    | 'PI_4'
    | '1_PI'
    | '2_PI'
    | '2_SQRTPI'
    | 'SQRT2'
    | 'SQRT1_2'
    | 'INFINITY'
    | 'NAN'
    ;

// constant
//     : CONSTANT_E        # ConstE
//     | CONSTANT_LOG2E    # ConstLog2e
//     | CONSTANT_LOG10E   # ConstLog10e
//     | CONSTANT_LN2      # ConstLn2
//     | CONSTANT_LN10     # ConstLn10
//     | CONSTANT_PI       # ConstPi
//     | CONSTANT_PI_2     # ConstPi_2
//     | CONSTANT_PI_4     # ConstPi_4
//     | CONSTANT_1_PI     # Const1_Pi
//     | CONSTANT_2_PI     # Const2_Pi
//     | CONSTANT_2_SQRTPI # Const2_SqrtPi
//     | CONSTANT_SQRT2    # ConstSqrt2
//     | CONSTANT_SQRT1_2  # ConstSqrt1_2
//     | CONSTANT_INFINITY # ConstInfinity
//     | CONSTANT_NAN      # ConstNan
//     ;

// CONSTANT_E : E ;
// CONSTANT_LOG2E : L O G '2' E ;
// CONSTANT_LOG10E : L O G '1' '0' E ;
// CONSTANT_LN2 : L N '2' ;
// CONSTANT_LN10 : L N '1' '0' ;
// CONSTANT_PI : P I ;
// CONSTANT_PI_2 : P I '_' '2' ;
// CONSTANT_PI_4 : P I '_' '4' ;
// CONSTANT_1_PI : '1' '_' P I ;
// CONSTANT_2_PI : '2' '_' P I ;
// CONSTANT_2_SQRTPI : '2' '_' S Q R T P I ;
// CONSTANT_SQRT2 : S Q R T '2' ;
// CONSTANT_SQRT1_2 : S Q R T '1' '_' '2' ;
// CONSTANT_INFINITY : I N F I N I T Y ;
// CONSTANT_NAN : N A N ;

SYMBOL : [a-zA-Z~!@$%^&*_\-+=<>.?/:] [a-zA-Z0-9~!@$%^&*_\-+=<>.?/:]* ;

NUMBER : [-+]? [0-9]+ ('.' [0-9]+)? ('e' [-+]?[0-9]+)? ;

STRING : '"' ([\u0020-\u0021\u0023-\u005b\u005d-\u007e] | '\\' [\u0022\u005c])+? '"' ;

WS : [ \t\n\r]+ -> skip ;

// fragment A : [aA] ;
// fragment B : [bB] ;
// fragment C : [cC] ;
// fragment D : [dD] ;
// fragment E : [eE] ;
// fragment F : [fF] ;
// fragment G : [gG] ;
// fragment H : [hH] ;
// fragment I : [iI] ;
// fragment J : [jJ] ;
// fragment K : [kK] ;
// fragment L : [lL] ;
// fragment M : [mM] ;
// fragment N : [nN] ;
// fragment O : [oO] ;
// fragment P : [pP] ;
// fragment Q : [qQ] ;
// fragment R : [rR] ;
// fragment S : [sS] ;
// fragment T : [tT] ;
// fragment U : [uU] ;
// fragment V : [vV] ;
// fragment W : [wW] ;
// fragment X : [xX] ;
// fragment Y : [yY] ;
// fragment Z : [zZ] ;
