# Makefile for libnetwork
# Dependencies: libdata libui

CC = g++
CXX = g++
CFLAGS = -Wall -Wextra -O2 -fPIC
CXXFLAGS = $(CFLAGS) -std=c++11
LDFLAGS = -shared
AR = ar
ARFLAGS = rcs

# Library paths
LIB_DIR = ../
LIBDATA_DIR = $(LIB_DIR)/libdata
LIBUI_DIR = $(LIB_DIR)/libui

# Include paths
INCLUDES = -I./include -I$(LIBDATA_DIR)/include -I$(LIBUI_DIR)/include

# Library linking
LIBS = -L$(LIBDATA_DIR) -ldata -L$(LIBUI_DIR) -lui

# Source files
SOURCES = $(wildcard src/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

# Target library name
TARGET_SHARED = libnetwork.so
TARGET_STATIC = libnetwork.a

# Default target
all: $(TARGET_SHARED) $(TARGET_STATIC)

# Build shared library
$(TARGET_SHARED): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

# Build static library
$(TARGET_STATIC): $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $^

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET_SHARED) $(TARGET_STATIC)

# Phony targets
.PHONY: all clean