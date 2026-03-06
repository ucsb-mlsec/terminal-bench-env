# Makefile for libcore
# Dependencies: libutils libdata

CC = g++
CFLAGS = -Wall -Wextra -O2 -fPIC -Iinclude
LDFLAGS = -shared
AR = ar
ARFLAGS = rcs

LIB_NAME = libcore
STATIC_LIB = $(LIB_NAME).a
SHARED_LIB = $(LIB_NAME).so

SRC_DIR = src
OBJ_DIR = obj
INC_DIR = include

SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

LIB_DIRS = -L../libutils -L../libdata
LIBS = -lutils -ldata

all: $(STATIC_LIB) $(SHARED_LIB)

$(STATIC_LIB): $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $^

$(SHARED_LIB): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIB_DIRS) $(LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(STATIC_LIB) $(SHARED_LIB)

.PHONY: all clean