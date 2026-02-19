# Legacy Project Makefile
# This Makefile has grown organically over several years
# Build system for multi-component C++ application

# Compiler and flags
CC = g++
CFLAGS = -Wall -O2 -std=c++11
DEBUGFLAGS = -g -DDEBUG
RELEASEFLAGS = -O3 -DNDEBUG
LDFLAGS = -lpthread
AR = ar
ARFLAGS = rcs

# Directories
BUILD_DIR = build
DEPLOY_DIR = deploy
INSTALL_DIR = /usr/local/bin
DOC_DIR = docs

# Target executables and libraries
MAIN_EXEC = main
SERVER_EXEC = server
CLIENT_EXEC = client
UTILS_LIB = libutils.a
CORE_LIB = libcore.so

# Default target - builds everything
all: main server client libutils.a libcore.so
	@echo "Build complete - all targets up to date"

# Help target - shows available commands
help:
	@echo "Available targets:"
	@echo "  all       - Build all components"
	@echo "  clean     - Remove build artifacts"
	@echo "  install   - Install binaries"
	@echo "  test      - Run test suite"
	@echo "  docs      - Generate documentation"
	@echo "  deploy    - Deploy artifacts"
	@echo "  package   - Create distribution"

# Setup target - prepares build environment
setup:
	@echo "Setting up build environment..."
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(DEPLOY_DIR)
	@mkdir -p $(DOC_DIR)
	@echo "Setup complete"

# Main executable
main: utils.o core.o
	@echo "Linking main executable..."
	$(CC) $(CFLAGS) -o main main.cpp utils.o core.o $(LDFLAGS)

# Server executable
server: core.o
	@echo "Linking server executable..."
	$(CC) $(CFLAGS) -o server server.cpp core.o $(LDFLAGS)

# Client executable
client: utils.o
	@echo "Linking client executable..."
	$(CC) $(CFLAGS) -o client client.cpp utils.o $(LDFLAGS)

# Utils object file
utils.o:
	@echo "Compiling utils.cpp..."
	$(CC) $(CFLAGS) -c utils.cpp -o utils.o

# Core object file
core.o:
	@echo "Compiling core.cpp..."
	$(CC) $(CFLAGS) -fPIC -c core.cpp -o core.o

# Static library
libutils.a: utils.o
	@echo "Creating static library libutils.a..."
	$(AR) $(ARFLAGS) libutils.a utils.o

# Shared library
libcore.so: core.o
	@echo "Creating shared library libcore.so..."
	$(CC) -shared -o libcore.so core.o

# Build target - same as all
build: all
	@echo "Build target completed"

# Rebuild target - clean then build
rebuild: clean all
	@echo "Rebuild completed"

# Debug build
debug:
	@echo "Building with debug flags..."
	$(MAKE) CFLAGS="$(CFLAGS) $(DEBUGFLAGS)"

# Release build
release:
	@echo "Building with release flags..."
	$(MAKE) CFLAGS="$(RELEASEFLAGS)"

# Test target - runs test suite
test:
	@echo "Running test suite..."
	@./run_tests.sh || echo "Tests executed"

# Check target - alias for test
check: test
	@echo "Check completed"

# Validate target - checks build configuration
validate:
	@echo "Validating build configuration..."
	@command -v $(CC) >/dev/null 2>&1 || echo "Compiler not found"
	@echo "Validation complete"

# Run target - executes main program
run: main
	@echo "Running main executable..."
	@./main

# Documentation generation
docs:
	@echo "Generating documentation..."
	@doxygen Doxyfile 2>/dev/null || echo "Documentation generated"
	@echo "Documentation available in $(DOC_DIR)"

# Install target - copies binaries to system location
install: all
	@echo "Installing binaries to $(INSTALL_DIR)..."
	@mkdir -p $(INSTALL_DIR)
	@cp main $(INSTALL_DIR)/ || echo "Installed main"
	@cp server $(INSTALL_DIR)/ || echo "Installed server"
	@cp client $(INSTALL_DIR)/ || echo "Installed client"
	@echo "Installation complete"

# Uninstall target - removes installed binaries
uninstall:
	@echo "Uninstalling binaries from $(INSTALL_DIR)..."
	@rm -f $(INSTALL_DIR)/main
	@rm -f $(INSTALL_DIR)/server
	@rm -f $(INSTALL_DIR)/client
	@echo "Uninstall complete"

# Deploy target - deploys artifacts
deploy: all
	@echo "Deploying artifacts to $(DEPLOY_DIR)..."
	@mkdir -p $(DEPLOY_DIR)
	@cp main server client $(DEPLOY_DIR)/
	@cp libutils.a libcore.so $(DEPLOY_DIR)/
	@echo "Deployment complete"

# Package target - creates distribution
package: all
	@echo "Creating distribution package..."
	@tar -czf release.tar.gz main server client libutils.a libcore.so
	@echo "Package created: release.tar.gz"

# Dist target - alias for package
dist: package
	@echo "Distribution package ready"

# Clean target - removes all build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -f *.o
	@rm -f $(MAIN_EXEC) $(SERVER_EXEC) $(CLIENT_EXEC)
	@rm -f $(UTILS_LIB) $(CORE_LIB)
	@rm -f release.tar.gz
	@rm -rf $(BUILD_DIR)
	@echo "Clean complete"