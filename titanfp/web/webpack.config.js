const path = require('path');
const MonacoWebpackPlugin = require('monaco-editor-webpack-plugin');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const CopyWebpackPlugin = require('copy-webpack-plugin');
const { CleanWebpackPlugin } = require('clean-webpack-plugin');

module.exports = {
    entry: {
        layout: './layout.js',
        titanic: './titanfp.js',
    },
    output: {
        filename: '[name].bundle.js',
        path: path.resolve(__dirname, 'dist'),
    },
    optimization: {
        splitChunks: {
            chunks: 'all',
        },
    },
    module: {
        rules: [
            {
                test: /\.css$/,
                use: ['style-loader', 'css-loader'],
            },
        ],
    },
    plugins: [
        new MonacoWebpackPlugin(),
        new HtmlWebpackPlugin({
            template: 'index.html',
            filename: 'index.html',
            inject: false,
        }),
        new HtmlWebpackPlugin({
            template: 'evaluate.html',
            filename: 'evaluate.html',
            inject: 'body',
            favicon: 'favicon.ico',
        }),
        new HtmlWebpackPlugin({
            template: 'imgeval.html',
            filename: 'imgeval.html',
            inject: 'body',
            favicon: 'favicon.ico',
        }),
        new CopyWebpackPlugin({
            patterns: [
		{from : 'titanfp.css'},
            	{from : 'piceberg.png'},
            	{from : 'piceberg_round.png'},
	    ]
        }),
        new CleanWebpackPlugin(),
    ],
};
