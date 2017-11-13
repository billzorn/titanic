const $ = require("jquery");
const antlr4 = require("antlr4");
const FPCoreLexer = require("./gen/FPCoreLexer.js").FPCoreLexer;
const FPCoreParser = require("./gen/FPCoreParser.js").FPCoreParser;
const FPCoreVisitor = require("./gen/FPCoreVisitor.js").FPCoreVisitor;
const ast = require("./fpcast.js");
const ops = ast.operations;

class Visitor extends FPCoreVisitor {
    visitParse(ctx) {
        const cores = [];
        for (const child of ctx.children) {
            const parsed = child.accept(this);
            if (parsed) {
                cores.push(parsed);
            }
        }
        return cores;
    }

    visitFpcore(ctx) {
        const input_vars = ctx.inputs.map(x => x.text);
        const props = {};
        for (const child of ctx.props) {
            const [name, x] = child.accept(this);
            props[name] = x;
        }
        const e = ctx.e.accept(this);
        const core = {
            args: input_vars,
            properties: props,
            name: props[":name"],
            pre: props[":pre"],
            e: e,
        };
        return core;
    }

    visitFpimp(ctx) {
        throw new Error("unsupported: FPImp");
    }

    visitExprNum(ctx) {
        return new ast.Val(ctx.c.text);
    }

    visitExprConst(ctx) {
        return new ast.Val(ctx.c.text);
    }

    visitExprVar(ctx) {
        return new ast.Var(ctx.x.text);
    }

    visitExprUnop(ctx) {
        const op = ctx.op.getText();
        if (op === "-") {
            return new ops["neg"](ctx.arg0.accept(this));
        } else {
            return new ops[op](ctx.arg0.accept(this));
        }
    }

    visitExprBinop(ctx) {
        const op = ctx.op.getText();
        return new ops[op](ctx.arg0.accept(this), ctx.arg1.accept(this));
    }

    visitExprComp(ctx) {
        const op = ctx.op.getText();
        const visitor = this;
        return new ops[op](...(function*() {
            for (const e of ctx.args) {
                yield e.accept(visitor);
            }
        })());
    }

    visitExprLogical(ctx) {
        const op = ctx.op.getText();
        const visitor = this;
        return new ops[op](...(function*() {
            for (const e of ctx.args) {
                yield e.accept(visitor);
            }
        })());
    }

    visitExprIf(ctx) {
        throw new Error("unsupported: If");
    }

    visitExprLet(ctx) {
        throw new Error("unsupported: Let");
    }

    visitExprWhile(ctx) {
        throw new Error("unsupported: While");
    }

    visitPropStr(ctx) {
        return [ctx.name.text, ctx.s.text.slice(1, -1)];
    }

    visitPropList(ctx) {
        return [ctx.name.text, ctx.syms.map(x => x.text)];
    }

    visitPropExpr(ctx) {
        return [ctx.name.text, ctx.e.accept(this)];
    }
}

function parse(input) {
    const input_stream = new antlr4.InputStream(input);
    const lexer = new FPCoreLexer(input_stream);
    const token_stream  = new antlr4.CommonTokenStream(lexer);
    const parser = new FPCoreParser(token_stream);
    const parse_tree = parser.parse();
    return [parser, parse_tree];
}

function compile(input) {
    const [parser, tree] = parse(input);
    visitor = new Visitor();
    return visitor.visit(tree);
}

// web interface

function onCore() {
    const root = parentForm(this);
    const userCore = $(".incore", root).val();
    if (userCore) {
        const compiledCore = compile(userCore)[0];
        let coreStr = compiledCore.e.toString();
        if (compiledCore.pre) {
            coreStr = "pre: " + compiledCore.pre.toString() + "\n" + coreStr;
        }
        coreStr = "FPCore (" + compiledCore.args.join(" ") + ")\n" + coreStr;
        $(".outcore", root).html(coreStr);
        root.data("core", compiledCore);
        onArgs.call(this);
    }
}

function onArgs() {
    const root = parentForm(this);
    const compiledCore = root.data("core");
    if (compiledCore) {
        const userArgs = $(".arguments", root).val();
        const args = userArgs.split(";").map(x => x.trim());
        const ctx = {};
        for (let i = 0; i < compiledCore.args.length; i++) {
            ctx[compiledCore.args[i]] = args[i];
        }
        let resultStr = "result: " + compiledCore.e.apply(ctx);
        if (compiledCore.pre) {
            resultStr = "pre: " + compiledCore.pre.apply(ctx) + "\n" + resultStr;
        }
        $(".outval", root).html(resultStr);
    }
}

// set up the web interface

function parentForm(element) {
    return $(element).closest(".fpc");
}

function setupCore(rootClass) {
    const root = $(rootClass);
    $(".incore", root).on("input", onCore);
    $(".arguments", root).on("input", onArgs);
    onCore.call($(".incore", root).get(0));
}

exports.setupCore = setupCore;


//testing
// let inputDoc = "";

// process.stdin.setEncoding("utf8");

// process.stdin.on("readable", () => {
//     const chunk = process.stdin.read();
//     if (chunk) {
//         inputDoc += chunk;
//     }
// });

// process.stdin.on("end", () => {
//     const cores = compile(inputDoc);
//     console.log(cores[0]);
//     console.log(cores[0].e.apply({"x": 0.125}));
// });
