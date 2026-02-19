# Makefile for dataproc binary
# This configuration is optimized for minimal binary size

CC=gcc

# Aggressive optimization flags for size reduction
# -O3: Maximum optimization
# -s: Strip all symbol table and relocation information
# -DNDEBUG: Disable assertions
# -fomit-frame-pointer: Don't keep frame pointer for functions that don't need one
# -ffunction-sections: Place each function in its own section
# -fdata-sections: Place each data item in its own section
CFLAGS=-O3 -s -DNDEBUG -fomit-frame-pointer -ffunction-sections -fdata-sections

# Linker flags for additional size optimization
# --gc-sections: Remove unused sections
# --strip-all: Remove all symbol information
LDFLAGS=-Wl,--gc-sections,--strip-all

SRC_DIR=src
BIN_DIR=bin

SOURCES=$(SRC_DIR)/main.c $(SRC_DIR)/validator.c

TARGET=$(BIN_DIR)/dataproc

all: $(TARGET)

$(TARGET): $(SOURCES)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) $(LDFLAGS)
	@echo "Binary size: $$(du -h $(TARGET) | cut -f1)"

clean:
	rm -f $(TARGET)

.PHONY: all clean