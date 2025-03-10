import $ from 'jquery';
// import debounce from 'lodash-es/debounce';

// import * as monaco from 'monaco-editor';
import * as monaco from 'monaco-editor/esm/vs/editor/editor.api';

import {editor_box, resize_webtool, do_layout,
        analysis_up, analysis_right, analysis_down} from './layout';


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
// const editor_val = getFPCore();
const editor_val = '';


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
        'enable_analysis': $('#enable-analysis').is(':checked'),
        'heatmap': $('#heatmap').is(':checked'),
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
    if (state.hasOwnProperty('enable_analysis')) {
        $('#enable-analysis').prop('checked', state.enable_analysis == 'true');
    }
    if (state.hasOwnProperty('heatmap')) {
        $('#heatmap').prop('checked', state.heatmap == 'true');
    }
}


// submission logic

let result_id = 1;

function scroll_output() {
    const output_div = $('#output');
    output_div.scrollTop(output_div.prop("scrollHeight"));
}

function show_report(result_body, report) {
    $('#overlay-result').html(result_body);
    const report_row = '<div class="output-row"> <pre class="output-pre code">' + report + '</pre></div>';
    $('#overlay-report').html(report_row);
    analysis_up();
}

function register_result() {
    const result_id_name = 'result-' + result_id;
    result_id += 1;

    const output_div = $('#output');

    output_div.append(
        '<div id="' + result_id_name + '" class="output-item"></div>'
    );

    // // this refuses to scroll past the top of the new element
    // // ... because the div is 0px tall, until the content is updated
    // const result_element = document.getElementById(result_id_name);
    // result_element.scrollIntoView({block: "end"});

    return result_id_name;
}

function eval_result(result, id_name, link_url) {
    let body = '';
    let body_for_report = '';

    if (result.success) {
        body += '<div class="output-row">';
        body += '<p class="code">(<a target="_blank" rel="noopener noreferrer" href="' + link_url + '">link</a>)</p>';
        for (let [k, v] of result.args) {
            body += '<p>' + k + ' = ' + v + '</p>';
        }
        if (result.pre_val == 'False') {
            body += '<p class="text">(precondition not satisfied)</p>';
        }
        body += '</div>';
        body += '<div class="output-row">';
        body += '<pre class="output-pre code">' + result.e_val + '</pre>';
        body += '</div>';
        if ('mat_2d' in result) {
            body += '<table class="output-row" style="border-spacing: 10px 0;">';
            for (let row of result.mat_2d) {
                body += '<tr>';
                for (let col of row) {
                    body += '<td><pre class="output-pre code">' + col + '</pre></td>';
                }
                body += '</tr>';
            }
            body += '</table>';
        }
        if ('result_img' in result) {
            body += '<div class="output-row">';
            body += '<img src="data:image/png;base64, ' + result.result_img + '" alt="computed image" />';
            body += '</div>';
        }
        body += '<div class="output-row">';
        if ('report' in result) {
            body_for_report += body;
            body_for_report += '</div>';

            const button_name = 'report-' + id_name;
            body += '<pre class="output-pre code">(' + result.eval_count + ' expressions evaluated) ';
            body += '<button id="' + button_name + '" class="text" onclick="">View report</button>';
            body += '</pre>';
        } else {
            body += '<pre class="output-pre code">(' + result.eval_count + ' expressions evaluated)</pre>';
        }
        body += '</div>';
    } else {
        if (result.message) {
            body = '<pre class="output-pre code">' + result.message + '</pre>';
        } else {
            body = '<p>Something went wrong, oops!</p>';
        }
    }

    $('#' + id_name).html(body);

    // need to wait to create the body before we can register listeners on the button
    if (result.success && 'report' in result) {
        const button_name = 'report-' + id_name;
        const report_html = result.report;
        $('#' + button_name).click(() => show_report(body_for_report, report_html));
    }

    scroll_output();
}

function submit_eval() {
    const data_object = get_webtool_state();
    const id_name = register_result();

    const u = new URL(window.location);
    const s = new URLSearchParams(data_object);
    u.search = s.toString();

    const link_url = u.toString();

    $('#' + id_name).html('<div class="output-row"><p class="code">'
                          + '(<a target="_blank" rel="noopener noreferrer" href="' + link_url + '">link</a>) Evaluating...</p>'
                          + '</div>');

    scroll_output();

    const img_input = $('#user_upload')[0];
    if (img_input.files && img_input.files[0]) {
        const file = img_input.files[0];

        var reader = new FileReader();

        reader.onload = function(e) {
            var data = e.target.result.replace("data:"+ file.type +";base64,", '');
            data_object['usr_img'] = data;

            const payload = JSON.stringify(data_object);
            $.ajax({
                type: 'POST',
                url: 'eval',
                data: payload,
                contentType: 'application/json',
                success: (result) => eval_result(result, id_name, link_url),
            });
        }

        reader.readAsDataURL(file);
    } else {
        const payload = JSON.stringify(data_object);
        $.ajax({
            type: 'POST',
            url: 'eval',
            data: payload,
            contentType: 'application/json',
            success: (result) => eval_result(result, id_name, link_url),
        });
    }
}

$('#evaluate').click(submit_eval);

// $('#control-analysis-up').click(analysis_up);
// $('#control-analysis-right').click(analysis_right);
$('#analysis-up').click(analysis_up);
$('#analysis-right').click(analysis_right);
$('#analysis-down').click(analysis_down);


// image input

function read_img() {
    const img_input = this;
    if (img_input.files && img_input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            $('#user_preview')
                .attr('src', e.target.result);
        };

        reader.readAsDataURL(img_input.files[0]);
    }
}

$('#user_upload').change(read_img);


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
    $('#permalink').prop('href', create_permalink());
}

$('#permalink').on('mouseenter', on_link);

restore_from_permalink();
