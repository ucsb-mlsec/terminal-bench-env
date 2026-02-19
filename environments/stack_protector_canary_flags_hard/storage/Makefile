# Makefile for storage module
# Data storage and persistence layer

CC = gcc
AR = ar
ARFLAGS = rcs

# Compiler flags - optimization and debugging
CFLAGS = -O2 -g -Wall -Wextra -pedantic
CFLAGS += -Iinclude

# Additional flags for performance
CFLAGS += -march=native -mtune=native

# Library name
LIB = libstorage.a

# Source files
SRCS = storage.c
OBJS = $(SRCS:.c=.o)

# Build targets
all: $(LIB)

$(LIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

storage.o: storage.c
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(OBJS) $(LIB)

.PHONY: install
install: $(LIB)
	cp $(LIB) /usr/local/lib/
	cp storage.h /usr/local/include/