import * as monaco from 'monaco-editor';
// import * as monaco from 'monaco-editor/esm/vs/editor/editor.api';


// import {smt2} from './monaco_smt2';
// import {getSMT2} from './editor_demo';
// monaco.languages.register({id: 'smt2'});
// monaco.languages.setMonarchTokensProvider('smt2', smt2);
// const editor_lang = 'smt2';
// const editor_val = getSMT2();


// import {colortest} from './monaco_colortest';
// import {getColortest} from './editor_demo';
// monaco.languages.register({id: 'colortest'});
// monaco.languages.setMonarchTokensProvider('colortest', colortest);
// const editor_lang = 'colortest';
// const editor_val = getColortest();


import {fpcore} from './monaco_fpcore';
import {getFPCore} from './editor_demo';
monaco.languages.register({id: 'fpcore'});
monaco.languages.setMonarchTokensProvider('fpcore', fpcore);
const editor_lang = 'fpcore';
const editor_val = getFPCore();


monaco.languages.setLanguageConfiguration('fpcore', {
    'brackets': [['(', ')'], ['[', ']']],
    'comments': {'lineComment': ';', 'blockComment': ['#|', '|#']},
});

// const keywords = [
//     // keywords
//     'FPCore', 'if', 'let', 'while', 'digits', '!',
//     // operators - mathematical
//     '+', '-', '*', '/', 'fabs',
//     'fma', 'exp', 'exp2', 'expm1', 'log',
//     'log10', 'log2', 'log1p', 'pow', 'sqrt',
//     'cbrt', 'hypot', 'sin', 'cos', 'tan',
//     'asin', 'acos', 'atan', 'atan2', 'sinh',
//     'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
//     'erf', 'erfc', 'tgamma', 'lgamma', 'ceil',
//     'floor', 'fmod', 'remainder', 'fmax', 'fmin',
//     'fdim', 'copysign', 'trunc', 'round', 'nearbyint',
//     // testing
//     '<', '>', '<=', '>=', '==',
//     '!=', 'and', 'or', 'not', 'isfinite',
//     'isinf', 'isnan', 'isnormal', 'signbit',
//     // cast
//     'cast',
//     // constants - mathematical
//     'E', 'LOG2E', 'LOG10E', 'LN2', 'LN10',
//     'PI', 'PI_2', 'PI_4', 'M_1_PI', 'M_2_PI',
//     'M_2_SQRTPI', 'SQRT2', 'SQRT1_2', 'INFINITY', 'NAN',
//     // boolean
//     'TRUE', 'FALSE',
//     // properties
//     ':name', ':description', ':cite', ':precision', ':round',
//     ':pre', ':spec', ':math-library',
//     // precision shorthands
//     'binary16', 'binary32', 'binary64',
//     'posit8', 'posit16', 'posit32', 'posit64',
//     'real', 'integer',
//     // rounding modes
//     'nearestEven', 'nearestAway', 'toPositive', 'toNegatie', 'toZero',
// ];

// const suggestions = keywords.map(
//     kw => Object({
//         label: kw,
//         kind: monaco.languages.CompletionItemKind.Text,
//         insertText: kw,
//     })
// );

// Unfortunately, this works extremely poorly...
// monaco.languages.registerCompletionItemProvider('fpcore', {
//     provideCompletionItems: () => Object({suggestions: suggestions}),
// });


monaco.editor.create(document.getElementById('editor'), {
    language: editor_lang,
    value: editor_val,
});
