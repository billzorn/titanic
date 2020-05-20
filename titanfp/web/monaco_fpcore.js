// FPCore 1.1 languages
// See http://fpbench.org/spec/fpcore-1.1.html for more information
export const fpcore = {

    defaultToken: 'invalid',

    annotations: [
        '!', '#',
    ],

    keywords: [
        'FPCore', 'if', 'let', 'let*',
        'tensor', 'tensor*', 'for', 'for*', 'while', 'while*',
        'digits',
    ],

    operators: [
        // tensor
        'dim', 'size', 'ref',
        // mathematical
        '+', '-', '*', '/', 'fabs',
        'fma', 'exp', 'exp2', 'expm1', 'log',
        'log10', 'log2', 'log1p', 'pow', 'sqrt',
        'cbrt', 'hypot', 'sin', 'cos', 'tan',
        'asin', 'acos', 'atan', 'atan2', 'sinh',
        'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
        'erf', 'erfc', 'tgamma', 'lgamma', 'ceil',
        'floor', 'fmod', 'remainder', 'fmax', 'fmin',
        'fdim', 'copysign', 'trunc', 'round', 'nearbyint',
        // testing
        '<', '>', '<=', '>=', '==',
        '!=', 'and', 'or', 'not', 'isfinite',
        'isinf', 'isnan', 'isnormal', 'signbit',
        // cast
        'cast',
    ],

    constants: [
        // mathematical
        'E', 'LOG2E', 'LOG10E', 'LN2', 'LN10',
        'PI', 'PI_2', 'PI_4', 'M_1_PI', 'M_2_PI',
        'M_2_SQRTPI', 'SQRT2', 'SQRT1_2', 'INFINITY', 'NAN',
        // boolean
        'TRUE', 'FALSE',
        // let's stick tensor literals in here
        'array',
    ],

    brackets: [
        ['(', ')', 'delimiter.parenthesis'],
        ['[', ']', 'delimiter.square'],
    ],

    decnum: /[-+]?([0-9]+(\.[0-9]+)?|\.[0-9]+)(e[-+]?[0-9]+)?/,

    hexnum: /[+-]?0x([0-9a-f]+(\.[0-9a-f]+)?|\.[0-9a-f]+)(p[-+]?[0-9]+)?/,

    rational: /[+-]?[0-9]+\/[0-9]*[1-9][0-9]*/,

    symbol: /[a-zA-Z~!@$%^&*_\-+=<>.?\/:][a-zA-Z0-9~!@$%^&*_\-+=<>.?\/:]*/,

    escape: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

    tokenizer: {
        root: [
            // whitespace and brackets
            {include: '@whitespace'},
            [/[()\[\]]/, '@brackets'],

            // symbols, including keywords and operators
            [/@symbol/, {cases: {'@annotations': 'string',
                                 '@keywords': 'keyword',
                                 '@operators': 'keyword.operator',
                                 '@constants': 'number.constant',
                                 ':.+': 'string.property',
                                 '@default': 'identifier'}}],

            // number literals
            [/@decnum/, 'number.float'],
            [/@hexnum/, 'number.hex'],
            [/@rational/, 'number.float'],

            // strings
            [/"([^"\\]|\\.)*$/, 'string.invalid'],  // non-teminated
            [/"/, {token: 'string.quote', bracket: '@open', next: '@string'}],
        ],

        string: [
            [/[^\\"]+/, 'string'],
            [/@escape/, 'string.escape'],
            [/\\./, 'string.escape.invalid'],
            [/"/, {token: 'string.quote', bracket: '@close', next: '@pop'}],
        ],

        whitespace: [
            [/[ \t\r\n]+/, 'white'],
            // line comment
            [/;.*$/, 'comment'],
            // block comment (like racket)
            [/#\|/, 'comment', '@comment' ],
        ],

        // comments can nest!
        comment: [
            [/[^#|]+/, 'comment' ],
            [/#\|/, 'comment', '@push'],
            [/\|#/, 'comment', '@pop'],
            [/[#|]/, 'comment'],
        ],
    }

}
