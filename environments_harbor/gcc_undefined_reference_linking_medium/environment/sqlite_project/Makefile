CC = gcc
CFLAGS = -Wall -I/usr/include
TARGET = dbapp
OBJS = main.o db_utils.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET)

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET)