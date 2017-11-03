#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

(cd $DIR \
&& rm -rf gen_java/ \
&& java -cp antlr/antlr-4.7-complete.jar org.antlr.v4.Tool -o gen_java FPCore.g4 \
&& javac -cp antlr/antlr-4.7-complete.jar gen_java/FPCore*.java \
&& rm -rf gen/ \
&& java -cp antlr/antlr-4.7-complete.jar org.antlr.v4.Tool -Dlanguage=Python3 -visitor -no-listener -o gen FPCore.g4 \
&& rm -rf www/gen/ \
&& java -cp antlr/antlr-4.7-complete.jar org.antlr.v4.Tool -Dlanguage=JavaScript -visitor -no-listener -o www/gen FPCore.g4)
