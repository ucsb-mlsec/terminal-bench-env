CC=gcc
CFLAGS=-O2
TARGET=calculator
SOURCES=main.c operations.c utils.c
OBJECTS=main.o operations.o utils.o

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)