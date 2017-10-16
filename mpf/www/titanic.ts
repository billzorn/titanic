function sExpr() {
    const e : any = document.getElementById("sExpr");
    return e;
}
function sLabel() {
    const e : any = document.getElementById("sLabel");
    return e;
}

function demoButton() {
    const e : any = document.getElementById("demoButton");
    return e;
}

function wBits() {
    const e : any = document.getElementById("wBits");
    return e;
}
function pBits() {
    const e : any = document.getElementById("pBits");
    return e;
}

function fmtBinary16() {
    const e : any = document.getElementById("fmtBinary16");
    return e;
}
function fmtBinary32() {
    const e : any = document.getElementById("fmtBinary32");
    return e;
}
function fmtBinary64() {
    const e : any = document.getElementById("fmtBinary64");
    return e;
}
function fmtCustom() {
    const e : any = document.getElementById("fmtCustom");
    return e;
}

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

function explainInput(explanation: string, valid: boolean) {
    let label = sLabel();
    label.innerHTML = "(" + explanation + ")";
    if (valid) {
        label.style.color = "Green";
        demoButton().disabled = false;
    } else {
        label.style.color = "Red";
        demoButton().disabled = true;
    }
}

const ieee_w_p = {
    16: [5, 11, "binary16"],
    32: [8, 24, "binary32"],
    64: [11, 53, "binary64"],
    128: [15, 113, "binary128"],
    256: [19, 237, "binary256"]
}

function explainBV(size: number) {
    const ew = wBits();
    const ep = pBits();
    const wIn : number = parseInt(ew.value, 10);
    const pIn : number = parseInt(ep.value, 10);
    const w_p = ieee_w_p[size];
    if (wIn + pIn == size) {
        if (w_p) {
            const [w, p, name] = w_p;
            if (w == wIn && p == pIn)
                explainInput("packed " + name, true);
            else
                explainInput("packed bitvector", true);
        } else
            explainInput("packed bitvector", true);
    } else if (w_p) {
        const [w, p, name] = w_p;
        explainInput("packed " + name, true);
    } else
        explainInput("bitvector: bad w and p", false);
}

function bvsize(s: string): number {
    const m = bv_re.exec(s);
    console.log(m);
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
    const m = input_re.exec(sExpr().value);
    if (!m) {
        explainInput("unknown format", false);
    } else if (m[2]) {
        explainInput("infinity", true);
    } else if (m[4]) {
        explainInput("not a number", true);
    } else if (m[8]) {
        explainInput("mathematical constant", true);
    } else if (m[10]) {
        if (!m[9] && !m[11] && nodot_re.test(m[10]))
            explainBV(m[10].length);
        else
            explainInput("binary number", true);
    } else if (m[13]) {
        if (!m[12] && !m[14] && nodot_re.test(m[13]))
            explainBV(m[13].length * 4);
        else
            explainInput("hexadecimal number", true);
    } else if (m[16]) {
        explainInput("decimal number", true);
    } else if (m[19]) {
        explainInput("ordinal", true);
    } else if (m[20]) {
        explainInput("rational number", true);
    } else if (m[23]) {
        explainInput("exponential notation", true);
    } else if (m[28]) {
        const w = bvsize(m[29]);
        const p = bvsize(m[30]) + 1;
        explainInput("implicit triple: w=" + w + ", p=" + p, true);
    }
    else if (m[31]) {
        const w = bvsize(m[32]);
        const p = bvsize(m[33]) + 1;
        explainInput("implicit triple: w=" + w + ", p=" + p, true);
    }
    else if (m[34]) {
        const w = bvsize(m[35]);
        const p = bvsize(m[37]) + 1;
        explainInput("explicit triple: w=" + w + ", p=" + p, true);
    }
    else if (m[38]) {
        const w = bvsize(m[35]);
        const p = bvsize(m[37]) + 1;
        explainInput("explicit triple: w=" + w + ", p=" + p, true);
    } else {
        explainInput("mystery format", true);
    }
}

function onWP() {
    const ew = wBits();
    const ep = pBits();
    if (ew.value == 5 && ep.value == 11) {
        fmtBinary16().checked = true;
    } else if (ew.value == 8 && ep.value == 24) {
        fmtBinary32().checked = true;
    } else if (ew.value == 11 && ep.value == 53) {
        fmtBinary64().checked = true;
    } else {
        fmtCustom().checked = true;
    }
    onText();
}

function onFormat(x : number) {
    const ew = wBits();
    const ep = pBits();
    if (x == 16) {
        ew.value = "5";
        ep.value = "11";
    } else if (x == 32) {
        ew.value = "8";
        ep.value = "24";
    } else if (x == 64) {
        ew.value = "11";
        ep.value = "53";
    }
}
