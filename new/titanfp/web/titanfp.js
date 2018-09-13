var ace = require('brace');
require('brace/theme/textmate');
require('brace/mode/scheme');
exports.ace = ace;

const $ = require('jquery');
exports.$ = $;

function handle_result(data) {
    // could parse some json or something
    $('.result').html(data);
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
    selected = $('#backend-select').val();
    $(".backend-descr").css("display", "none");
    $("#descr-" + selected).css("display", "");
}

function setup_page() {
    $("#backend-select").on("change", onBackend);
}

exports.submit_core = submit_core;
exports.setup_page = setup_page;
