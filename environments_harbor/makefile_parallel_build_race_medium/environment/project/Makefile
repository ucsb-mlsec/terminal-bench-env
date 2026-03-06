CC=gcc
CFLAGS=-Wall -g

OBJS=main.o utils.o processor.o handler.o
TARGET=program

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

utils.o: utils.c
	$(CC) $(CFLAGS) -c utils.c

processor.o: processor.c
	$(CC) $(CFLAGS) -c processor.c

handler.o: handler.c
	$(CC) $(CFLAGS) -c handler.c

generator: generator.c
	$(CC) $(CFLAGS) -o generator generator.c

generated.h: generator
	./generator > generated.h

clean:
	rm -f $(OBJS) $(TARGET) generator generated.h