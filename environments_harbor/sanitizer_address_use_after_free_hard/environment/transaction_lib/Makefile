CC = gcc
CFLAGS = -Wall -g
LDFLAGS = 

OBJS = transaction.o util.o test_transactions.o
TARGET = test

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

transaction.o: transaction.c transaction.h
	$(CC) $(CFLAGS) -c transaction.c

util.o: util.c util.h
	$(CC) $(CFLAGS) -c util.c

test_transactions.o: test_transactions.c transaction.h util.h
	$(CC) $(CFLAGS) -c test_transactions.c

clean:
	rm -f $(OBJS) $(TARGET)