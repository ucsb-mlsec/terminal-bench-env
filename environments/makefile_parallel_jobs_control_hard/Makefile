CC = g++
CFLAGS = -c
AR = ar rcs

.PHONY: all clean

all: codegen generated.h libmath.a app1 app2

codegen: codegen.cpp
	$(CC) codegen.cpp -o codegen

generated.h:
	./codegen

math.o: math.cpp
	$(CC) $(CFLAGS) math.cpp -o math.o

libmath.a: math.o
	$(AR) libmath.a math.o

app1: app1.cpp
	$(CC) app1.cpp libmath.a -o app1

app2: app2.cpp
	$(CC) app2.cpp libmath.a -o app2

clean:
	rm -f *.o *.a codegen generated.h app1 app2