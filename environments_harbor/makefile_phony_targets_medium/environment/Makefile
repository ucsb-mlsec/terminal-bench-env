CC=gcc
CFLAGS=-Wall -O2
BUILD_DIR=build
TARGET=textproc

all: $(TARGET)

build:
	mkdir -p $(BUILD_DIR)

$(TARGET): $(BUILD_DIR)/main.o $(BUILD_DIR)/parser.o $(BUILD_DIR)/utils.o
	$(CC) $(CFLAGS) -o $(TARGET) $(BUILD_DIR)/main.o $(BUILD_DIR)/parser.o $(BUILD_DIR)/utils.o

$(BUILD_DIR)/main.o: main.c | build
	$(CC) $(CFLAGS) -c main.c -o $(BUILD_DIR)/main.o

$(BUILD_DIR)/parser.o: parser.c | build
	$(CC) $(CFLAGS) -c parser.c -o $(BUILD_DIR)/parser.o

$(BUILD_DIR)/utils.o: utils.c | build
	$(CC) $(CFLAGS) -c utils.c -o $(BUILD_DIR)/utils.o

docs:
	doxygen Doxyfile

test: $(TARGET)
	./run_tests.sh

clean:
	rm -f $(BUILD_DIR)/*.o $(TARGET)

distclean: clean
	rm -rf $(BUILD_DIR) docs/

install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

help:
	@echo "Available targets:"
	@echo "  all       - Build the text processing utility (default)"
	@echo "  build     - Create the build directory"
	@echo "  docs      - Generate documentation using Doxygen"
	@echo "  test      - Run the test suite"
	@echo "  clean     - Remove object files and executable"
	@echo "  distclean - Remove all generated files and directories"
	@echo "  install   - Install the executable to /usr/local/bin"
	@echo "  help      - Display this help message"