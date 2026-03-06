# Makefile for mathlib project

CC = gcc
CFLAGS = -Wall -O2
SRC_DIR = src
OBJ_DIR = obj

SOURCES = $(wildcard $(SRC_DIR)/*.c)
OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SOURCES))

.PHONY: static shared clean

static: libmathlib.a

libmathlib.a: $(OBJECTS)
	ar rcs libmathlib.a $(OBJECTS)

shared: libmathlib.so

libmathlib.so: $(OBJECTS)
	$(CC) -shared -o libmathlib.so $(OBJECTS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR)
	rm -f *.a *.so