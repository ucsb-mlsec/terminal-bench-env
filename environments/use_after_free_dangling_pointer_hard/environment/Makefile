CC=gcc
CFLAGS=-Wall -g

all: log_analyzer

log_analyzer: log_parser.o stats.o
	$(CC) $(CFLAGS) -o log_analyzer log_parser.o stats.o

log_parser.o: log_parser.c log_parser.h stats.h
	$(CC) $(CFLAGS) -c log_parser.c

stats.o: stats.c stats.h log_parser.h
	$(CC) $(CFLAGS) -c stats.c

clean:
	rm -f *.o log_analyzer