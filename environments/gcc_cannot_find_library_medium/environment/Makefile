CC = gcc
CFLAGS = -Wall -Wextra
LDFLAGS = 

all: config_parser

config_parser: config_parser.c
	$(CC) $(CFLAGS) config_parser.c -o config_parser $(LDFLAGS)

clean:
	rm -f config_parser

.PHONY: all clean