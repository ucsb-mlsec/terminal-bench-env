CC = gcc

CFLAGS = -O2 -Wall \
         -Werror \
         -pedantic \
         -std=c11 \
         -D_FORTIFY_SOURCE=2 \
         -I../auth \
         -I../network \
         -I../storage

LDFLAGS = -L../auth -L../network -L../storage \
          -lauth -lnetwork -lstorage \
          -lpthread

TARGET = app
OBJS = main.o

all: $(TARGET)

main.o: main.c
	$(CC) $(CFLAGS) -c main.c -o main.o

$(TARGET): $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) -o $(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean