import $ from 'jquery';
import debounce from 'lodash-es/debounce';


export const editor_box = [];

const border_px = 16;
const min_width = 800;
const min_height = 480;

export function resize_webtool() {
    const vw = Math.max(window.innerWidth, min_width);
    const vh = Math.max(window.innerHeight, min_height);
    const w = (vw - (border_px * 3)) / 2;
    const h = vh - (border_px * 2);

    $('#webtool').css({
        width: vw + 'px',
        height: vh + 'px',
    });
    $('.left').css({
        // no left border; monaco can put line numbers over there
        width: (w + border_px) + 'px',
        height:  h + 'px',
        left: '0px',
        top: border_px + 'px',
    });
    $('.right').css({
        width: w  + 'px',
        height:  h + 'px',
        left: (w + (border_px * 2)) + 'px',
        top: border_px + 'px',
    });

    if (editor_box.length > 0) {
        editor_box[0].layout({width: w + border_px, height: h});
    }
}

resize_webtool();

$(window).on('resize', debounce(resize_webtool, 100));


function on_backend() {
    const selected = $('#backend').val();
    switch (selected) {
    case 'mpmf':
        $('#mpmf-opts').css('display', '');
        $('#float-opts').css('display', 'none');
        $('#posit-opts').css('display', 'none');
        break;
    case 'ieee754':
    case 'softfloat':
    case 'sinking-point':
        $('#mpmf-opts').css('display', 'none');
        $('#float-opts').css('display', '');
        $('#posit-opts').css('display', 'none');
        break;
    case 'posit':
    case 'softposit':
    case 'sinking-posit':
        $('#mpmf-opts').css('display', 'none');
        $('#float-opts').css('display', 'none');
        $('#posit-opts').css('display', '');
        break;
    }
}

function on_float_prec() {
    const w = $('#float-w').val();
    const p = $('#float-p').val();

    if (w == 5 && p == 11) {
        $('#float-select').val('binary16');
    } else if (w == 8 && p == 24) {
        $('#float-select').val('binary32');
    } else if (w == 11 && p == 53) {
        $('#float-select').val('binary64');
    } else {
        $('#float-select').val('custom');
    }
}

function on_float_prec_select() {
    const selected = $('#float-select').val();
    switch (selected) {
    case 'binary16':
        $('#float-w').val(5);
        $('#float-p').val(11);
        break;
    case 'binary32':
        $('#float-w').val(8);
        $('#float-p').val(24);
        break;
    case 'binary64':
        $('#float-w').val(11);
        $('#float-p').val(53);
        break;
    }
}

function on_posit_prec() {
    const es = $('#posit-es').val();
    const nbits = $('#posit-nbits').val();

    if (es == 0 && nbits == 8) {
        $('#posit-select').val('posit8');
    } else if (es == 1 && nbits == 16) {
        $('#posit-select').val('posit16');
    } else if (es == 2 && nbits == 32) {
        $('#posit-select').val('posit32');
    } else if (es == 4 && nbits == 64) {
        $('#posit-select').val('posit64');
    } else {
        $('#posit-select').val('custom');
    }
}

function on_posit_prec_select() {
    const selected = $('#posit-select').val();
    switch (selected) {
    case 'posit8':
        $('#posit-es').val(0);
        $('#posit-nbits').val(8);
        break;
    case 'posit16':
        $('#posit-es').val(1);
        $('#posit-nbits').val(16);
        break;
    case 'posit32':
        $('#posit-es').val(2);
        $('#posit-nbits').val(32);
        break;
    case 'posit64':
        $('#posit-es').val(4);
        $('#posit-nbits').val(64);
        break;
    }
}

export function do_layout() {
    on_backend();
    on_float_prec();
    on_posit_prec();
}

do_layout();

$('#backend').on('change', on_backend);

$('#float-w').on('change', on_float_prec);
$('#float-p').on('change', on_float_prec);
$('#float-select').on('change', on_float_prec_select);

$('#posit-es').on('change', on_posit_prec);
$('#posit-nbits').on('change', on_posit_prec);
$('#posit-select').on('change', on_posit_prec_select);
