// FPCore 1.1 languages
// See http://fpbench.org/spec/fpcore-1.1.html for more information
export const colortest = {

    defaultToken: '',

    tokenizer: {
        root: [
            [/<!DOCTYPE/, 'metatag', '@doctype'],

            [/^identifier$/, 'identifier'],
            [/^entity$/, 'entity'],
            [/^constructor$/, 'constructor'],
            [/^operator$/, 'operator'],
            [/^operators$/, 'operators'],
            [/^tag$/, 'tag'],
            [/^namespace$/, 'namespace'],
            [/^keyword$/, 'keyword'],
            [/^info-token$/, 'info-token'],
            [/^type$/, 'type'],
            [/^string$/, 'string'],
            [/^warn-token$/, 'warn-token'],
            [/^predefined$/, 'predefined'],
            [/^string.escape$/, 'string.escape'],
            [/^error-token$/, 'error-token'],
            [/^invalid$/, 'invalid'],
            [/^comment$/, 'comment'],
            [/^debug-token$/, 'debug-token'],
            [/^comment.doc$/, 'comment.doc'],
            [/^regexp$/, 'regexp'],
            [/^constant$/, 'constant'],
            [/^attribute$/, 'attribute'],

            [/^delimiter.curly$/, 'delimiter.curly'],
            [/^delimiter.square$/, 'delimiter.square'],
            [/^delimiter.parenthesis$/, 'delimiter.parenthesis'],
            [/^delimiter.angle$/, 'delimiter.angle'],
            [/^delimiter.array$/, 'delimiter.array'],
            [/^delimiter.bracket$/, 'delimiter.bracket'],
            [/^delimiter$/, 'delimiter'],
            [/^number.hex$/, 'number.hex'],
            [/^number.octal$/, 'number.octal'],
            [/^number.binary$/, 'number.binary'],
            [/^number.float$/, 'number.float'],
            [/^number.constant$/, 'number.constant'],
            [/^number$/, 'number'],
            [/^variable.name$/, 'variable.name'],
            [/^variable.value$/, 'variable.value'],
            [/^variable$/, 'variable'],
            [/^meta.content$/, 'meta.content'],
            [/^meta$/, 'meta'],
            [/^metatag.content$/, 'metatag.content'],
            [/^metatag$/, 'metatag'],
            [/^attribute.name.html/, 'attribute.name.html'],
            [/^attribute.name/, 'attribute.name'],
            [/^attribute.foobar/, 'attribute.foobar'],

            {include: '@whitespace'},
        ],

        doctype: [
	    [/[^>]+/, 'metatag.content'],
	    [/>/, 'metatag', '@pop'],
	],

        whitespace: [
            [/[ \t\r\n]+/, 'white'],
            [/;.*$/,    'comment'],
        ],
    },

};
