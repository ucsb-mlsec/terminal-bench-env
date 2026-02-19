# Makefile for libdata
# Dependencies: libui libcore

CC = g++
CFLAGS = -Wall -Wextra -O2 -fPIC -Iinclude
LDFLAGS = -shared
AR = ar
ARFLAGS = rcs

LIB_NAME = libdata
STATIC_LIB = $(LIB_NAME).a
SHARED_LIB = $(LIB_NAME).so

SRC_DIR = src
OBJ_DIR = obj
INCLUDE_DIR = include

SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Dependency library paths
DEP_LIBS = -L../libui -L../libcore -lui -lcore

.PHONY: all clean static shared

all: static shared

static: $(STATIC_LIB)

shared: $(SHARED_LIB)

$(STATIC_LIB): $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $^

$(SHARED_LIB): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^ $(DEP_LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(STATIC_LIB) $(SHARED_LIB)

install: $(STATIC_LIB) $(SHARED_LIB)
	cp $(STATIC_LIB) /usr/local/lib/
	cp $(SHARED_LIB) /usr/local/lib/
	cp -r $(INCLUDE_DIR)/* /usr/local/include/