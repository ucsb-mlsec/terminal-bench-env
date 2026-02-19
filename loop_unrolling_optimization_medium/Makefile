CC=gcc
CFLAGS=-O0

all: blur_filter

blur_filter: blur_filter.c
	$(CC) $(CFLAGS) -o blur_filter blur_filter.c -lm

clean:
	rm -f blur_filter output.pgm