CC = gcc
CFLAGS = -fPIC -Wall -O2
LDFLAGS = -shared -ldl
TARGET = libhook.so
SRC_DIR = src
OBJ_DIR = obj

SOURCES = $(SRC_DIR)/hooks.c $(SRC_DIR)/hook_loader.c
OBJECTS = $(OBJ_DIR)/hooks.o $(OBJ_DIR)/hook_loader.o

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean