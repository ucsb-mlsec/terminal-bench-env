# Makefile for libutils
# Dependencies: libnetwork

CC = g++
CFLAGS = -Wall -Wextra -O2 -fPIC -Iinclude
LDFLAGS = -L../libnetwork -lnetwork
AR = ar
ARFLAGS = rcs

SRC_DIR = src
OBJ_DIR = obj
INCLUDE_DIR = include

SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

TARGET_LIB = libutils.so
TARGET_STATIC = libutils.a

.PHONY: all clean

all: $(TARGET_LIB)

$(TARGET_LIB): $(OBJECTS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)

$(TARGET_STATIC): $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(TARGET_LIB) $(TARGET_STATIC)

install: $(TARGET_LIB)
	cp $(TARGET_LIB) /usr/local/lib/
	cp -r $(INCLUDE_DIR)/* /usr/local/include/