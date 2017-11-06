const antlr4 = require('antlr4');
const FPCoreLexer = require('./gen/FPCoreLexer').FPCoreLexer;
const FPCoreParser = require('./gen/FPCoreParser').FPCoreParser;
const FPCoreVisitor = require('./gen/FPCoreVisitor').FPCoreVisitor;

// ES6

class Visitor extends FPCoreVisitor {
    visitExprNum(ctx) {
        console.log("number: " + ctx.c.text);
    }

    visitExprConst(ctx) {
        console.log("constant: " + ctx.c.text);
    }

    visitExprVar(ctx) {
        console.log("variable: " + ctx.x.text);
    }
}

// Object.setPrototypeOf(Visitor.prototype, FPCoreVisitor);




// classic

// function Visitor() {
//     FPCoreVisitor.call(this);
//     return this;
// }

// Visitor.prototype = Object.create(FPCoreVisitor.prototype);
// Visitor.prototype.constructor = Visitor;

// Visitor.prototype.visitExprNum = function(ctx) {
//     console.log("number: " + ctx.c.text);
// }
// Visitor.prototype.visitExprConst = function(ctx) {
//     console.log("constant: " + ctx.c.text);
// }
// Visitor.prototype.visitExprVar = function(ctx) {
//     console.log("variable: " + ctx.x.text);
// }




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

exports.compile = compile;


// testing
// let inputDoc = "";

// process.stdin.setEncoding("utf8");

// process.stdin.on("readable", () => {
//     const chunk = process.stdin.read();
//     if (chunk) {
//         inputDoc += chunk;
//     }
// });

// process.stdin.on("end", () => {
//     console.log(compile(inputDoc).toString());
// });
