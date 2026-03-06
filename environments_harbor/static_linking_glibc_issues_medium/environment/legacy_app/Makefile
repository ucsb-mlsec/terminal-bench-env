CC=gcc
CFLAGS=-Wall -O2
SRC=calculator.c

all: dynamic static

dynamic: $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o calculator_dynamic -lm

static: $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o calculator_static

clean:
	rm -f calculator_dynamic calculator_static

.PHONY: all dynamic static clean