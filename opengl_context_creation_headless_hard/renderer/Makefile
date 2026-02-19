CC = gcc
CFLAGS = -O2 -Wall -Wextra
LDFLAGS = -lm -lpthread
INCLUDES = -I/usr/include

.PHONY: all run clean

all: render_test

render_test: render_test.c
	$(CC) $(CFLAGS) $(INCLUDES) -o render_test render_test.c $(LDFLAGS)

run: render_test
	./render_test

clean:
	rm -f render_test
	rm -f output/*.ppm