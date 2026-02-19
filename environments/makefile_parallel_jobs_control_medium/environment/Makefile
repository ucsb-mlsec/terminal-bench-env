CC=gcc
CFLAGS=-Wall
AR=ar

all: libmath.a libutils.a calculator

libmath.a: math_ops.o
	$(AR) rcs libmath.a math_ops.o

libutils.a: utils.o
	$(AR) rcs libutils.a utils.o

calculator: calculator.o
	$(CC) calculator.o -L. -lutils -lmath -o calculator

math_ops.o: math_ops.c
	$(CC) $(CFLAGS) -c math_ops.c -o math_ops.o

utils.o: utils.c
	$(CC) $(CFLAGS) -c utils.c -o utils.o

calculator.o: calculator.c
	$(CC) $(CFLAGS) -c calculator.c -o calculator.o

clean:
	rm -f *.o *.a calculator