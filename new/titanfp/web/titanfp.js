var ace = require('brace');
require('brace/theme/textmate');
require('brace/mode/scheme');
exports.ace = ace;

const $ = require('jquery');
exports.$ = $;

function handle_result(data) {
    // could parse some json or something
    $('#result').html(data);
}

function submit_core(editor) {
    const payload = JSON.stringify({
        'core' : editor.getValue(),
        'inputs' : $('#args').val(),
        'backend' : $('#backend-select').val(),
    });

    $.ajax({
        type: 'POST',
        url: 'demo',
        data: payload,
        contentType: 'application/json',
        success: handle_result
    });
}

function onBackend(x) {
    const selected = $('#backend-select').val();
    switch (selected) {
    case 'native':
    case 'np':
    case 'softfloat':
    case 'softposit':
    case 'fpcore':
        $('#submit_btn').html('Interpret FPCore');
        break;
    case 'core2c':
    case 'core2js':
    case 'core2smtlib2':
        $('#submit_btn').html('Translate FPCore');
        break;
    case 'canonicalize':
        $('#submit_btn').html('Canonicalize FPCore');
        break;
    case 'condense':
        $('#submit_btn').html('Condense FPCore');
        break;
    case 'minimize':
        $('#submit_btn').html('Minimize FPCore');
        break;
    default:
        $('#submit_btn').html('unknown backend');
    }
}

function setup_page() {
    $("#backend-select").on("change", onBackend);
}

exports.submit_core = submit_core;
exports.setup_page = setup_page;
