CC=gcc
CFLAGS=-O3 -Wall -Wextra -fPIC -std=c99
LDFLAGS=-shared

SOURCES=matrix_ops.c vector_ops.c fft.c linear_algebra.c utils.c
OBJECTS=$(SOURCES:.c=.o)
LIBRARY=libmathlib.so
STATIC_LIB=libmathlib.a

all: $(LIBRARY) $(STATIC_LIB)

$(LIBRARY): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^

$(STATIC_LIB): $(OBJECTS)
	ar rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

matrix_ops.o: matrix_ops.c matrix_ops.h
vector_ops.o: vector_ops.c vector_ops.h
fft.o: fft.c fft.h
linear_algebra.o: linear_algebra.c linear_algebra.h
utils.o: utils.c utils.h

clean:
	rm -f $(OBJECTS) $(LIBRARY) $(STATIC_LIB)

.PHONY: all clean