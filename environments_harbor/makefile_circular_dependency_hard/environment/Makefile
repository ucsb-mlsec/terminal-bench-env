PYTHON = python3
MODULES_DIR = modules
BUILD_DIR = build
DIST_DIR = dist
MODULES = alpha beta gamma delta epsilon

MODULE_FILES = $(addprefix $(MODULES_DIR)/, $(addsuffix .py, $(MODULES)))
PYC_FILES = $(addprefix $(BUILD_DIR)/, $(addsuffix .pyc, $(MODULES)))

.PHONY: all clean validate compile package test

# Cycle 1: all -> package -> compile -> all
all: validate compile package test

clean:
	rm -rf $(BUILD_DIR) $(DIST_DIR)
	mkdir -p $(BUILD_DIR) $(DIST_DIR)

# Cycle 2: validate -> build/validated.stamp -> compile -> validate
validate: $(BUILD_DIR)/validated.stamp

$(BUILD_DIR)/validated.stamp: compile
	@echo "Validating Python modules..."
	@for module in $(MODULE_FILES); do \
		$(PYTHON) -m py_compile $$module || exit 1; \
	done
	@touch $(BUILD_DIR)/validated.stamp

# Cycle 1 & 2: compile depends on both validate and all
compile: validate all $(BUILD_DIR)/compile.stamp

$(BUILD_DIR)/compile.stamp: package $(PYC_FILES)
	@echo "Compilation complete"
	@touch $(BUILD_DIR)/compile.stamp

# Cycle 3: alpha.pyc depends on beta.pyc and vice versa
$(BUILD_DIR)/alpha.pyc: $(MODULES_DIR)/alpha.py $(BUILD_DIR)/beta.pyc
	@echo "Compiling alpha module..."
	@mkdir -p $(BUILD_DIR)
	@$(PYTHON) -m py_compile $(MODULES_DIR)/alpha.py
	@cp $(MODULES_DIR)/__pycache__/alpha.*.pyc $(BUILD_DIR)/alpha.pyc 2>/dev/null || touch $(BUILD_DIR)/alpha.pyc

$(BUILD_DIR)/beta.pyc: $(MODULES_DIR)/beta.py $(BUILD_DIR)/alpha.pyc
	@echo "Compiling beta module..."
	@mkdir -p $(BUILD_DIR)
	@$(PYTHON) -m py_compile $(MODULES_DIR)/beta.py
	@cp $(MODULES_DIR)/__pycache__/beta.*.pyc $(BUILD_DIR)/beta.pyc 2>/dev/null || touch $(BUILD_DIR)/beta.pyc

$(BUILD_DIR)/gamma.pyc: $(MODULES_DIR)/gamma.py $(BUILD_DIR)/delta.pyc
	@echo "Compiling gamma module..."
	@mkdir -p $(BUILD_DIR)
	@$(PYTHON) -m py_compile $(MODULES_DIR)/gamma.py
	@cp $(MODULES_DIR)/__pycache__/gamma.*.pyc $(BUILD_DIR)/gamma.pyc 2>/dev/null || touch $(BUILD_DIR)/gamma.pyc

$(BUILD_DIR)/delta.pyc: $(MODULES_DIR)/delta.py $(BUILD_DIR)/epsilon.pyc
	@echo "Compiling delta module..."
	@mkdir -p $(BUILD_DIR)
	@$(PYTHON) -m py_compile $(MODULES_DIR)/delta.py
	@cp $(MODULES_DIR)/__pycache__/delta.*.pyc $(BUILD_DIR)/delta.pyc 2>/dev/null || touch $(BUILD_DIR)/delta.pyc

$(BUILD_DIR)/epsilon.pyc: $(MODULES_DIR)/epsilon.py $(BUILD_DIR)/gamma.pyc
	@echo "Compiling epsilon module..."
	@mkdir -p $(BUILD_DIR)
	@$(PYTHON) -m py_compile $(MODULES_DIR)/epsilon.py
	@cp $(MODULES_DIR)/__pycache__/epsilon.*.pyc $(BUILD_DIR)/epsilon.pyc 2>/dev/null || touch $(BUILD_DIR)/epsilon.pyc

# Cycle 4: test -> dist/package.tar.gz -> test
# Cycle 5: package -> build/compile.stamp -> package
package: $(DIST_DIR)/package.tar.gz

$(DIST_DIR)/package.tar.gz: $(BUILD_DIR)/compile.stamp test
	@echo "Creating distribution package..."
	@mkdir -p $(DIST_DIR)
	@tar czf $(DIST_DIR)/package.tar.gz -C $(MODULES_DIR) $(addsuffix .py, $(MODULES))
	@echo "Package created at $(DIST_DIR)/package.tar.gz"

# Cycle 4: test depends on dist/package.tar.gz which depends on test
test: $(DIST_DIR)/package.tar.gz $(BUILD_DIR)/test.stamp

$(BUILD_DIR)/test.stamp: $(PYC_FILES)
	@echo "Running import tests..."
	@for module in $(MODULES); do \
		echo "Testing import of $$module..."; \
		$(PYTHON) -c "import sys; sys.path.insert(0, '$(MODULES_DIR)'); import $$module" || exit 1; \
	done
	@touch $(BUILD_DIR)/test.stamp

# Additional cycle dependencies
$(BUILD_DIR)/validated.stamp: $(BUILD_DIR)/test.stamp

$(BUILD_DIR)/compile.stamp: $(BUILD_DIR)/validated.stamp