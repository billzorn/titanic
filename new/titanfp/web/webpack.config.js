const path = require('path');
const MonacoWebpackPlugin = require('monaco-editor-webpack-plugin');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const CleanWebpackPlugin = require('clean-webpack-plugin');

module.exports = {
    mode: 'development',
    entry: {
        app: './titanfp.js',
    },
    output: {
        filename: '[name].bundle.js',
        path: path.resolve(__dirname, 'dist'),
    },
    module: {
        rules: [{
            test: /\.css$/,
            use: ['style-loader', 'css-loader'],
        }],
    },
    plugins: [
        new MonacoWebpackPlugin(),
        new HtmlWebpackPlugin(),
        new CleanWebpackPlugin(['dist']),
    ],
};
