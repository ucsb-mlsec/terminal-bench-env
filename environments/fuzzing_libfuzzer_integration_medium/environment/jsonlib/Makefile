# Makefile for JSON parsing library

CC = clang
AR = ar
CFLAGS = -Wall -Wextra -O2 -g
ARFLAGS = rcs

LIB = libjson.a
OBJS = json_parser.o

.PHONY: all clean

all: $(LIB)

$(LIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

json_parser.o: json_parser.c json_parser.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(LIB)