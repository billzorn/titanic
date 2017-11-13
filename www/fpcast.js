// binary floating point conversion based on typed arrays

function hexToDouble(hexString) {
    const m = hexString.match(/^(?:0x)?([\da-f]{16})$/i);
    if (m === null) {
        throw new Error("invalid hex string for encoded double");
    } else {
        const hexChars = m[1];
        const buf = new ArrayBuffer(8);
        const uint8View = new Uint8Array(buf, 0, 8);
        const doubleView = new Float64Array(buf, 0, 1);
        let i = 7;
        for (const b of hexChars.match(/[\da-f]{2}/gi).map(x => parseInt(x, 16))) {
            uint8View[i] = b;
            i--;
        }
        return doubleView[0];
    }
}

// named mathematical constants

const constants = {
    "e": Math.E,
    "log2e": Math.LOG2E,
    "log10e": Math.LOG10E,
    "ln2": Math.LN2,
    "ln10": Math.LN10,
    "pi": Math.PI,
    "pi_2": hexToDouble("0x3ff921fb54442d18"), // 1.5707963267948966
    "pi_4": hexToDouble("0x3fe921fb54442d18"), // 0.7853981633974483
    "1_pi": hexToDouble("0x3fd45f306dc9c883"), // 0.3183098861837907
    "2_pi": hexToDouble("0x3fe45f306dc9c883"), // 0.6366197723675814
    "2_sqrtpi": hexToDouble("0x3ff20dd750429b6d"), // 1.1283791670955126
    "sqrt2": Math.SQRT2,
    "sqrt1_2": Math.SQRT1_2,
}

// pairwise implementations of n-ary comparisons

function implByPairs(op, conj) {
    function impl(...args) {
        if (args.length > 2) {
            return conj(...(function*() {
                for (let i = 0; i < args.length-1; i++) {
                    yield op(args[i], args[i+1]);
                }
            })());
        } else {
            return op(...args);
        }
    }
    return impl;
}

function implAllPairs(op, conj) {
    function impl(...args) {
        if (args.length > 2) {
            return conj(...(function*() {
                for (let i = 0; i < args.length-1; i++) {
                    for (let j = i+1; j < args.length; j++) {
                        yield op(args[i], args[j]);
                    }
                }
            })());
        } else {
            return op(...args);
        }
    }
    return impl;
}

function every(...arr) {
    return arr.every(x => x);
}

function some(...arr) {
    return arr.some(x => x);
}

// main expression class

class Expr {
    constructor(...args) {
        this.data = args;
        if (this.constructor.nargs !== null) {
            this.assert_nargs(this.constructor.nargs);
        }
    }

    assert_nargs(n) {
        if (this.data.length !== n) {
            throw new Error("expecting " + n + " arguments, got " + this.data);
        }
    }

    apply(ctx) {
        return this.constructor.op(...this.data.map(e => e.apply(ctx)));
    }

    toString() {
        return "(" + this.constructor.type + " " + this.data.map(e => e.toString()).join(" ") + ")";
    }
}
Expr.type = "Expr";
Expr.nargs = null;

// base case values and variables

class Val extends Expr {
    constructor(...args) {
        super(...args);
        this.data = this.data[0];
    }

    apply(ctx) {
        const c = constants[this.data];
        if (c === undefined) {
            return parseFloat(this.data);
        } else {
            return c;
        }
    }

    toString() {
        return this.data;
    }
}
Val.type = "Val";
Val.nargs = 1;

class Var extends Val {
    apply(ctx) {
        return ctx[this.data];
    }
}
Var.type = "Var";

// arithmetic

class Add extends Expr {}
Add.type = "+";
Add.nargs = 2;
Add.op = (x, y) => x + y;

class Sub extends Expr {}
Sub.type = "-";
Sub.nargs = 2;
Sub.op = (x, y) => x - y;

class Mul extends Expr {}
Mul.type = "*";
Mul.nargs = 2;
Mul.op = (x, y) => x * y;

class Div extends Expr {}
Div.type = "/";
Div.nargs = 2;
Div.op = (x, y) => x / y;

class Sqrt extends Expr {}
Sqrt.type = "sqrt";
Sqrt.nargs = 1;
Sqrt.op = (x) => Math.sqrt(x);

class Neg extends Expr {}
Neg.type = "neg";
Neg.nargs = 1;
Neg.op = (x) => -x;

// comparison

class LT extends Expr {}
LT.type = "<";
LT.op = implByPairs((x, y) => x < y, every);

class GT extends Expr {}
GT.type = ">";
GT.op = implByPairs((x, y) => x > y, every);

class LEQ extends Expr {}
LEQ.type = "<=";
LEQ.op = implByPairs((x, y) => x <= y, every);

class GEQ extends Expr {}
GEQ.type = ">=";
GEQ.op = implByPairs((x, y) => x >= y, every);

class EQ extends Expr {}
EQ.type = "==";
EQ.op = implByPairs((x, y) => x == y, every);

class NEQ extends Expr {}
NEQ.type = "!=";
NEQ.op = implByPairs((x, y) => x != y, every);

// logic

class And extends Expr {}
And.type = "and";
And.op = every;

class Or extends Expr {}
Or.type = "or";
Or.op = some;

class Not extends Expr {}
Not.type = "not";
Not.nargs = 1;
Not.op = (x) => !x;

// table of operations for the parser
const operations = {
    [Add.type]  : Add,
    [Sub.type]  : Sub,
    [Mul.type]  : Mul,
    [Div.type]  : Div,
    [Sqrt.type] : Sqrt,
    [Neg.type]  : Neg,
    [LT.type]   : LT,
    [GT.type]   : GT,
    [LEQ.type]  : LEQ,
    [GEQ.type]  : GEQ,
    [EQ.type]   : EQ,
    [NEQ.type]  : NEQ,
    [And.type]  : And,
    [Or.type]   : Or,
    [Not.type]  : Not,
}

// export interface

exports.Val = Val;
exports.Var = Var;
exports.operations = operations;
