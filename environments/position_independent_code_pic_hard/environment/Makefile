.PHONY: all clean

CC = gcc
CFLAGS = -Wall -O2

all: libdataproc.so

libdataproc.so: dataproc.o utils.o
	$(CC) -shared -o libdataproc.so dataproc.o utils.o

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *.so