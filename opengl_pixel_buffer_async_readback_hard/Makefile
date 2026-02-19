CC=gcc
CFLAGS=-O2 -Wall -Wextra -std=c99
LDFLAGS=-lGL -lGLU -lglut -lpthread
INCLUDES=-I./include

TARGET=capture_test
SRC_DIR=src
OBJ_DIR=obj

SOURCES=$(SRC_DIR)/capture_system.c
OBJECTS=$(OBJ_DIR)/capture_system.o
HEADERS=include/capture_system.h

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS)
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean