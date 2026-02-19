CC = gcc
CFLAGS = -Wall -O2

SOURCES = add.c subtract.c multiply.c divide.c
OBJECTS = add.o subtract.o multiply.o divide.o

STATIC_LIB = libmathutils.a
SHARED_LIB = libmathutils.so

all: $(STATIC_LIB) $(SHARED_LIB)

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

$(STATIC_LIB): $(OBJECTS)
	ar rcs $@ $(OBJECTS)

$(SHARED_LIB): $(OBJECTS)
	$(CC) -shared -o $@ $(OBJECTS)

clean:
	rm -f $(OBJECTS) $(STATIC_LIB) $(SHARED_LIB)

.PHONY: all clean