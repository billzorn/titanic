# download an ANTLR4 jar from:
#   http://www.antlr.org/download.html
# requires java to be available on the path. openjdk version "1.8.0_151" works.

ANTLR_JAR = antlr-4.9.1-complete.jar
PYTHON_TARGET_DIR = ..
RM = rm -f

all: fpcore fpy

fpcore:
	java -cp $(ANTLR_JAR) org.antlr.v4.Tool -Dlanguage=Python3 -visitor -no-listener -o $(PYTHON_TARGET_DIR) FPCore.g4

fpy:
	java -cp $(ANTLR_JAR) org.antlr.v4.Tool -Dlanguage=Python3 -visitor -no-listener -o $(PYTHON_TARGET_DIR) FPY.g4

clean:
	$(RM) $(PYTHON_TARGET_DIR)/FPCoreLexer.py
	$(RM) $(PYTHON_TARGET_DIR)/FPCoreParser.py
	$(RM) $(PYTHON_TARGET_DIR)/FPCoreVisitor.py
	$(RM) $(PYTHON_TARGET_DIR)/FPCoreLexer.tokens
	$(RM) $(PYTHON_TARGET_DIR)/FPCore.tokens
	$(RM) $(PYTHON_TARGET_DIR)/FPCoreLexer.interp
	$(RM) $(PYTHON_TARGET_DIR)/FPCore.interp
	$(RM) $(PYTHON_TARGET_DIR)/FPYLexer.py
	$(RM) $(PYTHON_TARGET_DIR)/FPYParser.py
	$(RM) $(PYTHON_TARGET_DIR)/FPYVisitor.py
	$(RM) $(PYTHON_TARGET_DIR)/FPYLexer.tokens
	$(RM) $(PYTHON_TARGET_DIR)/FPY.tokens
	$(RM) $(PYTHON_TARGET_DIR)/FPYLexer.interp
	$(RM) $(PYTHON_TARGET_DIR)/FPY.interp

.PHONY: fpcore fpy clean
