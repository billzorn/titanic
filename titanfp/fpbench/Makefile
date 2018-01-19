PYTHON_TARGET_DIR = fpbench

python_target:
	java -cp antlr/antlr-4.7-complete.jar org.antlr.v4.Tool -Dlanguage=Python3 -visitor -no-listener -o $(PYTHON_TARGET_DIR) FPCore.g4

clean:
	rm $(PYTHON_TARGET_DIR)/FPCoreLexer.py
	rm $(PYTHON_TARGET_DIR)/FPCoreParser.py
	rm $(PYTHON_TARGET_DIR)/FPCoreVisitor.py
	rm $(PYTHON_TARGET_DIR)/FPCoreLexer.tokens
	rm $(PYTHON_TARGET_DIR)/FPCore.tokens

.PHONY: python_target clean
