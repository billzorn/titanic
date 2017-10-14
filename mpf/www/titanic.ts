function sExpr() {
    const e : any = document.getElementById("sExpr");
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
    + "|\\(\\s*([0-9]*[1-9][0-9]*)\\s*(?:\\*\\*|\\^)\\s*([-+]?[0-9]+)\\s*\\))"
// 28
const re_tup3 = "(?:(?:fp\\s*[,\\s])?\\s*(0|1|(?:0b|\\#b)0|(?:0b|\\#b)1)\\s*[,\\s]\\s*((?:0b|\\#b)[0-1]+|(?:0x|\\#x)[0-9a-f]+)\\s*[,\\s]\\s*((?:0b|\\#b)[0-1]+|(?:0x|\\#x)[0-9a-f]+)"
    + "|\\(\\s*(?:fp\\s*[,\\s])?\\s*(0|1|(?:0b|\\#b)0|(?:0b|\\#b)1)\\s*[,\\s]\\s*((?:0b|\\#b)[0-1]+|(?:0x|\\#x)[0-9a-f]+)\\s*[,\\s]\\s*((?:0b|\\#b)[0-1]+|(?:0x|\\#x)[0-9a-f]+)\\s*\\))"
// 34
const re_tup4 = "(?:(?:fp\\s*[,\\s])?\\s*(0|1|(?:0b|\\#b)0|(?:0b|\\#b)1)\\s*[,\\s]\\s*((?:0b|\\#b)[0-1]+|(?:0x|\\#x)[0-9a-f]+)\\s*[,\\s]\\s*(0|1|(?:0b|\\#b)0|(?:0b|\\#b)1)\\s*[,\\s]\\s*((?:0b|\\#b)[0-1]+|(?:0x|\\#x)[0-9a-f]+)"
    + "|\\(\\s*(?:fp\\s*[,\\s])?\\s*(0|1|(?:0b|\\#b)0|(?:0b|\\#b)1)\\s*[,\\s]\\s*((?:0b|\\#b)[0-1]+|(?:0x|\\#x)[0-9a-f]+)\\s*[,\\s]\\s*(0|1|(?:0b|\\#b)0|(?:0b|\\#b)1)\\s*[,\\s]\\s*((?:0b|\\#b)[0-1]+|(?:0x|\\#x)[0-9a-f]+)\\s*\\))"

const bv_re = new RegExp("(0|1)|(?:0b|\\#b)([0-1]+)|(?:0x|\\#x)([0-9a-f]+)", "i")

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
        + ")\\s*$", "i")

function onText() {
    const m = input_re.exec(sExpr().value);
    let s = "";

    if (!m)
        s += "no match";
    else if (m[2])
        s += "infinity"
    else if (m[4])
        s += "not a number"
    else if (m[8])
        s += "mathematical constant"
    else if (m[10])
        s += "binary number"
    else if (m[13])
        s += "hexadecimal number"
    else if (m[16])
        s += "decimal number"
    else if (m[19])
        s += "ordinal"
    else if (m[20])
        s += "rational number"
    else if (m[23])
        s += "exponential notation"
    else if (m[28] || m[31])
        s += "implicit triple"
    else if (m[34] || m[38])
        s += "explicit triple"
    else
        s += "wut"

    console.log(s + ": " + sExpr().value);
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
