# Makefile for SPIN model verification
# Token Ring Protocol Verification

# Variables
MODEL = token_ring.pml
PAN = pan
SPIN = spin

# Phony targets
.PHONY: all verify clean check

# Default target
all: verify

# Main verification target
verify: $(MODEL)
	@echo "Compiling model..."
	$(SPIN) -a $(MODEL)
	@echo "Building verifier..."
	gcc -o $(PAN) pan.c -DMEMLIM=2048 -O2
	@echo "Running verification..."
	./$(PAN) -m10000
	@echo "Verification complete."

# Clean generated files
clean:
	@echo "Cleaning up generated files..."
	rm -f $(PAN) pan.c pan.h pan.t pan.m pan.b *.trail _spin_nvr.tmp
	@echo "Clean complete."

# Thorough verification check
check: $(MODEL)
	@echo "Compiling model for thorough check..."
	$(SPIN) -a $(MODEL)
	@echo "Building verifier with additional options..."
	gcc -o $(PAN) pan.c -DMEMLIM=2048 -O2
	@echo "Running thorough verification..."
	./$(PAN) -a -f -m100000
	@echo "Thorough verification complete."