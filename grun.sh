#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

(cd $DIR/gen_java \
&& java -cp ".:../antlr/antlr-4.7-complete.jar" org.antlr.v4.gui.TestRig "$@")
