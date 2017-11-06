const reparse = require("./reparse.js");
const fpcparser = require("./fpcparser.js");

exports.onText = reparse.onText;
exports.onWP = reparse.onWP;
exports.onFormat = reparse.onFormat;
exports.compileCore = fpcparser.compile;
