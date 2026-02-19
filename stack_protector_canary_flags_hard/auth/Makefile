CC = gcc
CFLAGS = -O2 -Wall -Wextra -pedantic
AR = ar
ARFLAGS = rcs

TARGET = libauth.a
OBJS = auth.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

auth.o: auth.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean