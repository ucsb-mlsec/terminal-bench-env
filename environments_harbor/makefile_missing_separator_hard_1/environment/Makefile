CC = gcc
CFLAGS = -Wall -Wextra -std=c99
TARGET = calculator

all: $(TARGET)

calculator.o  parser.o formatter.o
    $(CC) $(CFLAGS) -c calculator.c -o calculator.o

parser.o: parser.c
        $(CC) $(CFLAGS) -c parser.c -o parser.o

formatter.o: formatter.c
    $(CC) $(CFLAGS) -c formatter.c -o formatter.o

$(TARGET): calculator.o parser.o  formatter.o
        $(CC) calculator.o parser.o formatter.o -o $(TARGET)

clean:
        rm -f *.o $(TARGET)

.PHONY: all clean