import * as monaco from 'monaco-editor';
// or import * as monaco from 'monaco-editor/esm/vs/editor/editor.api';
// if shipping only a subset of the features & languages is desired

import {smt2} from './monaco_smt2';
import {colortest} from './monaco_colortest';
import {fpcore} from './monaco_fpcore';

monaco.languages.register({id: 'smt2'});
monaco.languages.register({id: 'colortest'});
monaco.languages.register({id: 'fpcore'});

monaco.languages.setMonarchTokensProvider('smt2', smt2);
monaco.languages.setMonarchTokensProvider('colortest', colortest);
monaco.languages.setMonarchTokensProvider('fpcore', fpcore);

monaco.languages.setLanguageConfiguration('fpcore', {
    'brackets': [['(', ')'], ['[', ']']],
    'comments': {'lineComment': ';', 'blockComment': ['#|', '|#']},
});

const keywords = [
    // keywords
    'FPCore', 'if', 'let', 'while', '!', 'digits',
    // operators - mathematical
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
    // constants - mathematical
    'E', 'LOG2E', 'LOG10E', 'LN2', 'LN10',
    'PI', 'PI_2', 'PI_4', 'M_1_PI', 'M_2_PI',
    'M_2_SQRTPI', 'SQRT2', 'SQRT1_2', 'INFINITY', 'NAN',
    // boolean
    'TRUE', 'FALSE',
];

const suggestions = keywords.map(
    kw => Object({
        label: kw,
        kind: monaco.languages.CompletionItemKind.Text,
        insertText: kw,
    })
);

monaco.languages.registerCompletionItemProvider('fpcore', {
    provideCompletionItems: () => Object({suggestions: suggestions}),
});

monaco.editor.create(document.getElementById('container'), {
    value: getFPCore(),
    language: 'fpcore',
});

function getSMT2() {
    return [
        '; This example illustrates different uses of the arrays',
        '; supported in Z3.',
        '; This includes Combinatory Array Logic (de Moura & Bjorner, FMCAD 2009).',
        ';',
        '(define-sort A () (Array Int Int))',
        '(declare-fun x () Int)',
        '(declare-fun y () Int)',
        '(declare-fun z () Int)',
        '(declare-fun a1 () A)',
        '(declare-fun a2 () A)',
        '(declare-fun a3 () A)',
        '(push) ; illustrate select-store',
        '(assert (= (select a1 x) x))',
        '(assert (= (store a1 x y) a1))',
        '(check-sat)',
        '(get-model)',
        '(assert (not (= x y)))',
        '(check-sat)',
        '(pop)',
        '(define-fun all1_array () A ((as const A) 1))',
        '(simplify (select all1_array x))',
        '(define-sort IntSet () (Array Int Bool))',
        '(declare-fun a () IntSet)',
        '(declare-fun b () IntSet)',
        '(declare-fun c () IntSet)',
        '(push) ; illustrate map',
        '(assert (not (= ((_ map and) a b) ((_ map not) ((_ map or) ((_ map not) b) ((_ map not) a))))))',
        '(check-sat)',
        '(pop)',
        '(push)',
        '(assert (and (select ((_ map and) a b) x) (not (select a x))))',
        '(check-sat)',
        '(pop)',
        '(push)',
        '(assert (and (select ((_ map or) a b) x) (not (select a x))))',
        '(check-sat)',
        '(get-model)',
        '(assert (and (not (select b x))))',
        '(check-sat)',
        ';; unsat, so there is no model.',
        '(pop)',
        '(push) ; illustrate default',
        '(assert (= (default a1) 1))',
        '(assert (not (= a1 ((as const A) 1))))',
        '(check-sat)',
        '(get-model)',
        '(assert (= (default a2) 1))',
        '(assert (not (= a1 a2)))',
        '(check-sat)',
        '(get-model)',
        '(pop)',
        '(exit)',
    ].join('\n');
}

function getFPCore() {
    return [
        '(FPCore (n)',
        ' :name "arclength"',
        ' :cite (precimonious-2013)',
        ' :precision binary64',
        ' :pre (>= n 0)',
        ' (let ([dppi (acos -1.0)])',
        '   (let ([h (/ dppi n)])',
        '     (while (<= i n)',
        '      ([s1',
        '        0.0',
        '        (let ([t2 (let ([x (* i h)])',
        '                    ;; inlined body of fun',
        '                    (while (<= k 5)',
        '                     ([d0',
        '                       (! :precision binary32 2.0)',
        '                       (! :precision binary32 (* 2.0 d0))]',
        '                      [t0',
        '                       x',
        '                       (+ t0 (/ (sin (* d0 x)) d0))]',
        '                      [k 1 (+ k 1)])',
        '                     t0))])',
        '          (let ([s0 (sqrt (+ (* h h) (* (- t2 t1) (- t2 t1))))])',
        '            (! :precision binary128 (+ s1 s0))))]',
        '       [t1',
        '        0.0',
        '        (let ([t2 (let ([x (* i h)])',
        '                    ;; inlined body of fun',
        '                    (while (<= k 5)',
        '                     ([d0',
        '                       (! :precision binary32 2.0)',
        '                       (! :precision binary32 (* 2.0 d0))]',
        '                      [t0',
        '                       x',
        '                       (+ t0 (/ (sin (* d0 x)) d0))]',
        '                      [k 1 (+ k 1)])',
        '                     t0))])',
        '          t2)]',
        '       [i',
        '        1',
        '        (+ i 1)])',
        '      s1))))',
    ].join('\n');
}

function getColortest() {
    return [
        '<!DOCTYPE foodoc>',
        '',
        'identifier',
        'entity',
        'constructor',
        'operator',
        'operators',
        'tag',
        'namespace',
        'keyword',
        'info-token',
        'type',
        'string',
        'warn-token',
        'predefined',
        'string.escape',
        'error-token',
        'invalid',
        'comment',
        'debug-token',
        'comment.doc',
        'regexp',
        'constant',
        'attribute',
        '',
        'delimiter.curly',
        'delimiter.square',
        'delimiter.parenthesis',
        'delimiter.angle',
        'delimiter.array',
        'delimiter.bracket',
        'delimiter',
        '',
        'number.hex',
        'number.octal',
        'number.binary',
        'number.float',
        'number.constant',
        'number',
        '',
        'variable.name',
        'variable.value',
        'variable',
        '',
        'meta.content',
        'meta',
        '',
        'metatag.content',
        'metatag',
        '',
        'attribute.name.html',
        'attribute.name',
        'attribute.foobar',
    ].join('\n');
}
