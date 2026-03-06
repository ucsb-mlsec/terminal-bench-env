CC = gcc
CFLAGS = -fPIC -Wall
LDFLAGS = -shared
BUILD_DIR = build
LIB_DIR = lib
TARGET = $(LIB_DIR)/libutils.so

SRCS = src/utils.c src/helpers.c src/math_ops.c
OBJS = $(BUILD_DIR)/utils.o $(BUILD_DIR)/helpers.o $(BUILD_DIR)/math_ops.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS) $(LIB_DIR)
	$(CC) $(LDFLAGS) -o $@ $(OBJS)

$(BUILD_DIR)/utils.o: src/utils.c $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/helpers.o: src/helpers.c $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/math_ops.o: src/math_ops.c $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIB_DIR):
	mkdir -p $(LIB_DIR)

clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR)