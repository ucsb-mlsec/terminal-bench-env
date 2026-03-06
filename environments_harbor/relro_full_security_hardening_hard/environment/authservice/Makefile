CC = gcc
CFLAGS = -Wall -O2
LDFLAGS = 

all: auth

auth: auth.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o auth auth.c

clean:
	rm -f auth *.o