const $ = require("jquery");

//  1
const re_inf = "([-+])?(inf|infinity|oo)";
//  3
const re_nan = "([-+])?(nan|snan)(?:([-+]?[0-9]+)|\\(\\s*([-+]?[0-9]+)\\s*\\))";
//  7
const re_fpc = "([-+])?(e|log2e|log10e|ln2|ln10|pi|pi_2|pi_4|2_sqrtpi|sqrt2|sqrt1_2)";
//  9
const re_bin = "([-+])?(?:0b|\\#b)([0-1]*\\.[0-1]+|[0-1]+\\.?)(?:[ep]([-+]?[0-9]+))?";
// 12
const re_hex = "([-+])?(?:0x|\\#x)([0-9a-f]*\\.[0-9a-f]+|[0-9a-f]+\\.?)(?:p([-+]?[0-9]+))?";
// 15
const re_dec = "([-+])?(?:0d|\\#d)?([0-9]*\\.[0-9]+|[0-9]+\\.?)(?:e([-+]?[0-9]+))?";
// 18
const re_ord = "([-+])?(?:0n|\\#n)([0-9]+)";
// 20
const re_rat = "([-+]?[0-9]+)\\s*\\/\\s*([-+]?[0-9]*[1-9][0-9]*)";
// 22
const re_exp = "([-+])?([0-9]*\\.[0-9]+|[0-9]+\\.?)\\s*\\*\\s*"
    + "(?:([0-9]*[1-9][0-9]*)\\s*(?:\\*\\*|\\^)\\s*([-+]?[0-9]+)"
    + "|\\(\\s*([0-9]*[1-9][0-9]*)\\s*(?:\\*\\*|\\^)\\s*([-+]?[0-9]+)\\s*\\))";
// 28
const re_tup3 = "(?:(?:fp\\s*[,\\s])?\\s*(0|1|(?:0b|\\#b)0|(?:0b|\\#b)1)\\s*[,\\s]\\s*((?:0b|\\#b)[0-1]+|(?:0x|\\#x)[0-9a-f]+)\\s*[,\\s]\\s*((?:0b|\\#b)[0-1]+|(?:0x|\\#x)[0-9a-f]+)"
    + "|\\(\\s*(?:fp\\s*[,\\s])?\\s*(0|1|(?:0b|\\#b)0|(?:0b|\\#b)1)\\s*[,\\s]\\s*((?:0b|\\#b)[0-1]+|(?:0x|\\#x)[0-9a-f]+)\\s*[,\\s]\\s*((?:0b|\\#b)[0-1]+|(?:0x|\\#x)[0-9a-f]+)\\s*\\))";
// 34
const re_tup4 = "(?:(?:fp\\s*[,\\s])?\\s*(0|1|(?:0b|\\#b)0|(?:0b|\\#b)1)\\s*[,\\s]\\s*((?:0b|\\#b)[0-1]+|(?:0x|\\#x)[0-9a-f]+)\\s*[,\\s]\\s*(0|1|(?:0b|\\#b)0|(?:0b|\\#b)1)\\s*[,\\s]\\s*((?:0b|\\#b)[0-1]+|(?:0x|\\#x)[0-9a-f]+)"
    + "|\\(\\s*(?:fp\\s*[,\\s])?\\s*(0|1|(?:0b|\\#b)0|(?:0b|\\#b)1)\\s*[,\\s]\\s*((?:0b|\\#b)[0-1]+|(?:0x|\\#x)[0-9a-f]+)\\s*[,\\s]\\s*(0|1|(?:0b|\\#b)0|(?:0b|\\#b)1)\\s*[,\\s]\\s*((?:0b|\\#b)[0-1]+|(?:0x|\\#x)[0-9a-f]+)\\s*\\))";
// 42

const bv_re = new RegExp("^(?:(0|1)|(?:0b|\\#b)([0-1]+)|(?:0x|\\#x)([0-9a-f]+))$", "i");
const nodot_re = new RegExp("^(?:[^\\.]+)$", "i");
const input_re = new RegExp(
    "^\\s*(?:" + [re_inf,
                  re_nan,
                  re_fpc,
                  re_bin,
                  re_hex,
                  re_dec,
                  re_ord,
                  re_rat,
                  re_exp,
                  re_tup3,
                  re_tup4].join("|")
        + ")\\s*$", "i");

function explainInput(explanation, valid) {
    const root = parentForm(this);
    const label = $(".slabel", root);
    const demoButton = $(".demobutton", root);
    label.html("(" + explanation + ")");
    if (valid) {
        label.css("color", "green");
        demoButton.prop("disabled", false);
    } else {
        label.css("color", "red");
        demoButton.prop("disabled", true);
    }
}

const ieee_w_p = {
    16: [5, 11, "binary16"],
    32: [8, 24, "binary32"],
    64: [11, 53, "binary64"],
    128: [15, 113, "binary128"],
    256: [19, 237, "binary256"],
};

function explainBV(size) {
    const root = parentForm(this);
    const wIn = parseInt($(".wbits", root).val(), 10);
    const pIn = parseInt($(".pbits", root).val(), 10);
    const w_p = ieee_w_p[size];
    if (wIn + pIn === size) {
        if (w_p) {
            const [w, p, name] = w_p;
            if (w === wIn && p === pIn) {
                explainInput.call(this, "packed " + name, true);
            } else {
                explainInput.call(this, "packed bitvector", true);
            }
        } else {
            explainInput.call(this, "packed bitvector", true);
        }
    } else if (w_p) {
        const [, , name] = w_p;
        explainInput.call(this, "packed " + name, true);
    } else {
        explainInput.call(this, "bitvector: bad w and p", false);
    }
}

function bvsize(s) {
    const m = bv_re.exec(s);
    if (!m) {
        return null;
    } else if (m[1]) {
        return 1;
    } else if (m[2]) {
        return m[2].length;
    } else if (m[3]) {
        return m[3].length * 4;
    }
}

function onText() {
    const root = parentForm(this);
    const m = input_re.exec($(".sexpr", root).val());
    if (!m) {
        explainInput.call(this, "unknown format", false);
    } else if (m[2]) {
        explainInput.call(this, "infinity", true);
    } else if (m[4]) {
        explainInput.call(this, "not a number", true);
    } else if (m[8]) {
        explainInput.call(this, "mathematical constant", true);
    } else if (m[10]) {
        if (!m[9] && !m[11] && nodot_re.test(m[10])) {
            explainBV.call(this, m[10].length);
        } else {
            explainInput.call(this, "binary number", true);
        }
    } else if (m[13]) {
        if (!m[12] && !m[14] && nodot_re.test(m[13])) {
            explainBV.call(this, m[13].length * 4);
        } else {
            explainInput.call(this, "hexadecimal number", true);
        }
    } else if (m[16]) {
        explainInput.call(this, "decimal number", true);
    } else if (m[19]) {
        explainInput.call(this, "ordinal", true);
    } else if (m[20]) {
        explainInput.call(this, "rational number", true);
    } else if (m[23]) {
        explainInput.call(this, "exponential notation", true);
    } else if (m[28]) {
        const w = bvsize(m[29]);
        const p = bvsize(m[30]) + 1;
        explainInput.call(this, "implicit triple: w=" + w + ", p=" + p, true);
    } else if (m[31]) {
        const w = bvsize(m[32]);
        const p = bvsize(m[33]) + 1;
        explainInput.call(this, "implicit triple: w=" + w + ", p=" + p, true);
    } else if (m[34]) {
        const w = bvsize(m[35]);
        const p = bvsize(m[37]) + 1;
        explainInput.call(this, "explicit triple: w=" + w + ", p=" + p, true);
    } else if (m[38]) {
        const w = bvsize(m[39]);
        const p = bvsize(m[41]) + 1;
        explainInput.call(this, "explicit triple: w=" + w + ", p=" + p, true);
    } else {
        explainInput.call(this, "mystery format", true);
    }
}

function onWP() {
    const root = parentForm(this);
    const w = parseInt($(".wbits", root).val(), 10);
    const p = parseInt($(".pbits", root).val(), 10);
    if (w === 5 && p === 11) {
        $(".fmt16", root).prop("checked", true);
    } else if (w === 8 && p === 24) {
        $(".fmt32", root).prop("checked", true);
    } else if (w === 11 && p === 53) {
        $(".fmt64", root).prop("checked", true);
    } else {
        $(".fmt0", root).prop("checked", true);
    }
    onText.call(this);
}

function onFormat(x) {
    const root = parentForm(this);
    const e = $(this);
    const ew = $(".wbits", root);
    const ep = $(".pbits", root);
    if (e.hasClass("fmt16")) {
        ew.val(5);
        ep.val(11);
    } else if (e.hasClass("fmt32")) {
        ew.val(8);
        ep.val(24);
    } else if (e.hasClass("fmt64")) {
        ew.val(11);
        ep.val(53);
    }
}

// exports.onText = onText;
// exports.onWP = onWP;
// exports.onFormat = onFormat;

function parentForm(element) {
    return $(element).closest(".demo");
}

function setupNumber(rootClass) {
    const root = $(rootClass);
    $(".sexpr", root).on("input", onText);
    $(".wp", root).on("input", onWP);
    $(".fmt", root).on("change", onFormat);
    onWP.call($(".wbits", root).get(0));
}

exports.setupNumber = setupNumber;
