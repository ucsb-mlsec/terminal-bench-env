# Legacy recursive makefile - DO NOT USE
# This demonstrates the problematic recursive make pattern

CC = gcc
CFLAGS = -Iinclude -Wall
SUBDIRS = parser formatter output

.PHONY: all subdirs clean

all: subdirs textutil

subdirs:
	@for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir; \
	done

textutil: main.o
	$(CC) -o textutil main.o parser/parser.o formatter/formatter.o output/output.o

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

clean:
	@for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done
	rm -f textutil *.o