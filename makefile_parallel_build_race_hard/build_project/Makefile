CC = gcc
CFLAGS = -Wall -O2
TARGET = main_program
GENERATED_SOURCES = generated1.c generated2.c
OBJECTS = main.o generated1.o generated2.o

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) -o $(TARGET) $(OBJECTS)

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

generated1.o: generated1.c
	$(CC) $(CFLAGS) -c generated1.c

generated2.o: generated2.c
	$(CC) $(CFLAGS) -c generated2.c

generated1.c:
	./codegen generated1

generated2.c:
	./codegen generated2

codegen: codegen.c
	$(CC) $(CFLAGS) -o codegen codegen.c

clean:
	rm -f *.o $(TARGET) codegen $(GENERATED_SOURCES)