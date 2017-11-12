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

class Expr {
    name = "Expr";
    nargs = null;
    data = [];
    op;

    constructor(...args) {
        this.data = args;
        if (this.nargs !== null) {
            this.assert_nargs(this.nargs);
        }
    }

    assert_nargs(n) {
        if (this.data.length !== n) {
            throw new Error("expecting " + n + " arguments, got " + this.data);
        }
    }

    apply(ctx) {
        return this.op(...this.data.map(e => e.apply(ctx)));
    }

    toString() {
        return "(" + this.name + this.data.map(e => e.toString()).join(" ") + ")";
    }
}

class Val extends Expr {
    name = "Val";
    nargs = 1;
    data = "";

    constructor(data) {
        this.data = data;
    }

    apply(ctx) {
        return parseFloat(this.data);
    }

    toString() {
        return this.data;
    }
}

class Var extends Val {
    name = "Var";

    apply(ctx) {
        return ctx[this.data];
    }
}

class Add extends Expr {
    name = "+";
    nargs = 2;
    op = (x, y) => x + y;
}

class Sub extends Expr {
    name = "-";
    nargs = 2;
    op = (x, y) => x - y;
}

class Mul extends Expr {
    name = "*";
    nargs = 2;
    op = (x, y) => x * y;
}

class Div extends Expr {
    name = "/";
    nargs = 2;
    op = (x, y) => x / y;
}

class Sqrt extends Expr {
    name = "sqrt";
    nargs = 1;
    op = (x) => Math.sqrt(x);
}

class Neg extends Expr {
    name = "neg";
    nargs = 1;
    op = (x) => -x;
}

class LT extends Expr {
    name = "<";
    op = implByPairs((x, y) => x < y, every);
}

class GT extends Expr {
    name = ">";
    op = implByPairs((x, y) => x > y, every);
}

class LEQ extends Expr {
    name = "<=";
    op = implByPairs((x, y) => x <= y, every);
}

class GEQ extends Expr {
    name = ">=";
    op = implByPairs((x, y) => x >= y, every);
}

class EQ extends Expr {
    name = "==";
    op = implByPairs((x, y) => x == y, every);
}

class NEQ extends Expr {
    name = "!=";
    op = implByPairs((x, y) => x != y, every);
}

class And extends Expr {
    name = "and";
    op = every;
}

class Or extends Expr {
    name = "or";
    op = some;
}

class Not extends Expr {
    name = "not";
    nargs = 1;
    op = (x) => !x;
}

exports.Val = Val;
exports.Var = Var;
exports.Add = Add;
exports.Sub = Sub;
exports.Mul = Mul;
exports.Div = Div;
exports.Sqrt = Sqrt;
exports.Neg = Neg;
exports.LT = LT;
exports.GT = GT;
exports.LEQ = LEQ;
exports.GEQ = GEQ;
exports.EQ = EQ;
exports.NEQ = NEQ;
exports.And = And;
exports.Or = Or;
exports.Not = Not;
