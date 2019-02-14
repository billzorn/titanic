import $ from 'jquery';
// import debounce from 'lodash-es/debounce';

// import * as monaco from 'monaco-editor';
import * as monaco from 'monaco-editor/esm/vs/editor/editor.api';

import {editor_box, resize_webtool, do_layout} from './layout';


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


const editor = monaco.editor.create(document.getElementById('editor'), {
    language: editor_lang,
    value: editor_val,
    scrollBeyondLastLine: false,
});

editor_box.push(editor);
resize_webtool();


function get_webtool_state() {
    return {
        'core': editor.getValue(),
        'args': $('#args').val(),
        'backend': $('#backend').val(),
        'w': $('#float-w').val(),
        'p': $('#float-p').val(),
        'float_override': $('#float-override').is(':checked'),
        'es': $('#posit-es').val(),
        'nbits': $('#posit-nbits').val(),
        'posit_override': $('#posit-override').is(':checked'),
    };
}

function set_webtool_state(state) {
    if (state.hasOwnProperty('core')) {
        editor.setValue(state.core);
    }
    if (state.hasOwnProperty('args')) {
        $('#args').val(state.args);
    }
    if (state.hasOwnProperty('backend')) {
        $('#backend').val(state.backend);
    }
    if (state.hasOwnProperty('w')) {
        $('#float-w').val(state.w);
    }
    if (state.hasOwnProperty('p')) {
        $('#float-p').val(state.p);
    }
    if (state.hasOwnProperty('float_override')) {
        $('#float-override').prop('checked', state.float_override == 'true');
    }
    if (state.hasOwnProperty('es')) {
        $('#posit-es').val(state.es);
    }
    if (state.hasOwnProperty('nbits')) {
        $('#posit-nbits').val(state.nbits);
    }
    if (state.hasOwnProperty('posit_override')) {
        $('#posit-override').prop('checked', state.posit_override == 'true');
    }
}


// submission logic

function eval_result() {
    console.log(result);
}

function submit_eval() {
    const payload = JSON.stringify(get_webtool_state());

    $.ajax({
        type: 'POST',
        url: 'test',
        data: payload,
        contentType: 'application/json',
        success: eval_result,
    });
}

$('#evaluate').click(submit_eval);


// permalinks

function create_permalink() {
    const state = get_webtool_state();

    const u = new URL(window.location);
    const s = new URLSearchParams(state);
    u.search = s.toString();

    return u.toString();
}

function restore_from_permalink() {
    const u = new URL(window.location);

    const state = {};
    for (let pair of u.searchParams.entries()) {
        state[pair[0]] = pair[1];
    }

    set_webtool_state(state);
    do_layout();
}

function on_link() {
    console.log('hi');
    $('#permalink').prop('href', create_permalink());
}

$('#permalink').on('mouseenter', on_link);

restore_from_permalink();
