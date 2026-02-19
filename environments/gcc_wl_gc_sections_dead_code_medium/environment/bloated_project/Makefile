CC=gcc
CFLAGS=-Wall -g

OBJS=util1.o util2.o util3.o util4.o util5.o util6.o util7.o util8.o main.o

all: program

program: $(OBJS)
	$(CC) $(CFLAGS) -o program $(OBJS)

util1.o: util1.c
	$(CC) $(CFLAGS) -c util1.c

util2.o: util2.c
	$(CC) $(CFLAGS) -c util2.c

util3.o: util3.c
	$(CC) $(CFLAGS) -c util3.c

util4.o: util4.c
	$(CC) $(CFLAGS) -c util4.c

util5.o: util5.c
	$(CC) $(CFLAGS) -c util5.c

util6.o: util6.c
	$(CC) $(CFLAGS) -c util6.c

util7.o: util7.c
	$(CC) $(CFLAGS) -c util7.c

util8.o: util8.c
	$(CC) $(CFLAGS) -c util8.c

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

clean:
	rm -f *.o program