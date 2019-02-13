const path = require('path');
const MonacoWebpackPlugin = require('monaco-editor-webpack-plugin');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const CleanWebpackPlugin = require('clean-webpack-plugin');
const CopyWebpackPlugin = require('copy-webpack-plugin');

module.exports = {
    mode: 'development',
    entry: {
        layout: './layout.js',
        titanic: './titanfp.js',
    },
    output: {
        filename: '[name].bundle.js',
        path: path.resolve(__dirname, 'dist'),
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
            favicon: 'favicon.ico',
        }),
        new CopyWebpackPlugin([
            'titanfp.css',
            'piceberg.png',
            'piceberg_round.png',
        ]),
        new CleanWebpackPlugin(['dist']),
    ],
};
