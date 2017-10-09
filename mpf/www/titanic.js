function sExpr() {
    const e = document.getElementById("sExpr");
    return e;
}
function wBits() {
    const e = document.getElementById("wBits");
    return e;
}
function pBits() {
    const e = document.getElementById("pBits");
    return e;
}
function fmtBinary16() {
    const e = document.getElementById("fmtBinary16");
    return e;
}
function fmtBinary32() {
    const e = document.getElementById("fmtBinary32");
    return e;
}
function fmtBinary64() {
    const e = document.getElementById("fmtBinary64");
    return e;
}
function fmtCustom() {
    const e = document.getElementById("fmtCustom");
    return e;
}
function onWP() {
    const ew = wBits();
    const ep = pBits();
    if (ew.value == 5 && ep.value == 11) {
        fmtBinary16().checked = true;
    }
    else if (ew.value == 8 && ep.value == 24) {
        fmtBinary32().checked = true;
    }
    else if (ew.value == 11 && ep.value == 53) {
        fmtBinary64().checked = true;
    }
    else {
        fmtCustom().checked = true;
    }
}
function onFormat(x) {
    const ew = wBits();
    const ep = pBits();
    if (x == 16) {
        ew.value = "5";
        ep.value = "11";
    }
    else if (x == 32) {
        ew.value = "8";
        ep.value = "24";
    }
    else if (x == 64) {
        ew.value = "11";
        ep.value = "53";
    }
}
