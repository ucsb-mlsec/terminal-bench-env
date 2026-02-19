CC = gcc
CFLAGS = -Wall
TARGET = app
OBJS = main.o utils.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

main.o: main.c
	$(CC) -c $(CFLAGS) main.c

utils.o: utils.c
	$(CC) -c $(CFLAGS) utils.c

clean:
	rm -f *.o $(TARGET)