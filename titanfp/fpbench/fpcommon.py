import re

from . import fpcast as ast

_int_re = re.compile(r'(?P<sgn>[-+]?)(?:(?P<dec>[0-9]+)|(?P<hex>0[xX][0-9A-Fa-f]+))')

def read_int(s):
    m = _int_re.fullmatch(s)
    if m:
        if m.group('dec'):
            return int(s, 10)
        else: # m.group('hex')
            return int(s, 16)
    else:
        return None


def _neg_or_sub(a, b=None):
    if b is None:
        return ast.Neg(a)
    else:
        return ast.Sub(a, b)

def _pos_or_add(a, b=None):
    if b is None:
        return a
    else:
        return ast.Add(a, b)

reserved_constructs = {
    # reserved
    'FPCore' : None,
    # annotations and special syntax
    '!' : None,
    '#' : None,
    'cast' : ast.Cast,
    'digits' : None,
    # control flow (these asts are assembled directly in the visitor)
    'if' : None,
    'let' : None,
    'let*' : None,
    'while' : None,
    'while*' : None,
    'for' : None,
    'for*' : None,
    'tensor' : None,
    'tensor*' : None,

    'abort' : ast.Abort,
    
    # arrays
    'array' : ast.Array,

    # tensor operations
    'dim' : ast.Dim,
    'size' : ast.Size,
    'ref' : ast.Ref,
    # IEEE 754 required arithmetic (negation is a special case of subtraction)
    '+' : _pos_or_add,
    '-' : _neg_or_sub,
    '*' : ast.Mul,
    '/' : ast.Div,
    '%' : ast.Fmod,
    'sqrt' : ast.Sqrt,
    'fma' : ast.Fma,
    # discrete operations
    'copysign' : ast.Copysign,
    'fabs' : ast.Fabs,
    # composite arithmetic
    'fdim' : ast.Fdim,
    'fmax' : ast.Fmax,
    'fmin' : ast.Fmin,
    'fmod' : ast.Fmod,
    'remainder' : ast.Remainder,
    # rounding and truncation
    'ceil' : ast.Ceil,
    'floor' : ast.Floor,
    'nearbyint' : ast.Nearbyint,
    'round' : ast.Round,
    'trunc' : ast.Trunc,
    # trig
    'acos' : ast.Acos,
    'acosh' : ast.Acosh,
    'asin' : ast.Asin,
    'asinh' : ast.Asinh,
    'atan' : ast.Atan,
    'atan2' : ast.Atan2,
    'atanh' : ast.Atanh,
    'cos' : ast.Cos,
    'cosh' : ast.Cosh,
    'sin' : ast.Sin,
    'sinh' : ast.Sinh,
    'tan' : ast.Tan,
    'tanh' : ast.Tanh,
    # exponentials
    'exp' : ast.Exp,
    'exp2' : ast.Exp2,
    'expm1' : ast.Expm1,
    'log' : ast.Log,
    'log10' : ast.Log10,
    'log1p' : ast.Log1p,
    'log2' : ast.Log2,
    # powers
    'cbrt' : ast.Cbrt,
    'hypot' : ast.Hypot,
    'pow' : ast.Pow,
    # other
    'erf' : ast.Erf,
    'erfc' : ast.Erfc,
    'lgamma' : ast.Lgamma,
    'tgamma' : ast.Tgamma,

    # comparison
    '<' : ast.LT,
    '>' : ast.GT,
    '<=' : ast.LEQ,
    '>=' : ast.GEQ,
    '==' : ast.EQ,
    '!=' : ast.NEQ,
    # testing
    'isfinite' : ast.Isfinite,
    'isinf' : ast.Isinf,
    'isnan' : ast.Isnan,
    'isnormal' : ast.Isnormal,
    'signbit' : ast.Signbit,
    # logic
    'and' : ast.And,
    'or' : ast.Or,
    'not' : ast.Not,
}

reserved_constants = {
    # mathematical constants
    'E' : ast.Constant('E'),
    'LOG2E' : ast.Constant('LOG2E'),
    'LOG10E' : ast.Constant('LOG10E'),
    'LN2' : ast.Constant('LN2'),
    'LN10' : ast.Constant('LN10'),
    'PI' : ast.Constant('PI'),
    'PI_2' : ast.Constant('PI_2'),
    'PI_4' : ast.Constant('PI_4'),
    'M_1_PI' : ast.Constant('M_1_PI'),
    'M_2_PI' : ast.Constant('M_2_PI'),
    'M_2_SQRTPI' : ast.Constant('M_2_SQRTPI'),
    'SQRT2' : ast.Constant('SQRT2'),
    'SQRT1_2' : ast.Constant('SQRT1_2'),
    # infinity and NaN
    'INFINITY' : ast.Constant('INFINITY'),
    'NAN' : ast.Constant('NAN'),
    # boolean constants
    'TRUE' : ast.Constant('TRUE'),
    'FALSE' : ast.Constant('FALSE'),
}


class FPCoreParserError(Exception):
    """Unable to parse FPCore."""


def sanitize_arglist(args, argnames=None):
    if argnames is None:
        argnames = set()
    else:
        argnames = set(argnames)
    for name, props, shape in args:
        if name in argnames:
            raise FPCoreParserError(f'duplicate argument name {name!s}')
        elif name in reserved_constants:
            raise FPCoreParserError(f'argument name {name!s} is a reserved constant')
        else:
            argnames.add(name)
        # Also check the names of dimensions.
        if shape:
            for dim in shape:
                if isinstance(dim, str):
                    if dim in argnames:
                        raise FPCoreParserError(f'duplicate argument name {dim!s} for tensor dimension')
                    elif dim in reserved_constants:
                        raise FPCoreParserError(f'dimension name {dim!s} is a reserved constant')
                    else:
                        argnames.add(dim)

_sym_re = re.compile(r'[a-zA-Z~!@$%^&*_\-+=<>.?/:][a-zA-Z0-9~!@$%^&*_\-+=<>.?/:]*')

def sanitize_symbol(s):
    if not _sym_re.fullmatch(s):
        raise FPCoreParserError(f'invalid symbol {s!s}')

_sym_re_simplified = re.compile(r'[a-zA-Z~@$^&_.?][a-zA-Z0-9~@$^&_.?]*')

def is_simple_symbol(s):
    return bool(_sym_re_simplified.fullmatch(s))
