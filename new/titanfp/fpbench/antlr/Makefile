# download an ANTLR4 jar from:
#   http://www.antlr.org/download.html
# requires java to be available on the path. openjdk version "1.8.0_151" works.

ANTLR_JAR = antlr-4.7.1-complete.jar
PYTHON_TARGET_DIR = ..

python_target:
	java -cp $(ANTLR_JAR) org.antlr.v4.Tool -Dlanguage=Python3 -visitor -no-listener -o $(PYTHON_TARGET_DIR) FPCore.g4

clean:
	rm $(PYTHON_TARGET_DIR)/FPCoreLexer.py
	rm $(PYTHON_TARGET_DIR)/FPCoreParser.py
	rm $(PYTHON_TARGET_DIR)/FPCoreVisitor.py
	rm $(PYTHON_TARGET_DIR)/FPCoreLexer.tokens
	rm $(PYTHON_TARGET_DIR)/FPCore.tokens
	rm $(PYTHON_TARGET_DIR)/FPCoreLexer.interp
	rm $(PYTHON_TARGET_DIR)/FPCore.interp

.PHONY: python_target clean
