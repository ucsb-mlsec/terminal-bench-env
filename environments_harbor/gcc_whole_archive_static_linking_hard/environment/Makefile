CC=gcc
CFLAGS=-Wall -O2
AR=ar

all: sensor_app

main.o: main.c
	$(CC) $(CFLAGS) -c main.c -o main.o

sensor_a.o: sensor_a.c
	$(CC) $(CFLAGS) -c sensor_a.c -o sensor_a.o

sensor_b.o: sensor_b.c
	$(CC) $(CFLAGS) -c sensor_b.c -o sensor_b.o

sensor_c.o: sensor_c.c
	$(CC) $(CFLAGS) -c sensor_c.c -o sensor_c.o

libsensor_a.a: sensor_a.o
	$(AR) rcs libsensor_a.a sensor_a.o

libsensor_b.a: sensor_b.o
	$(AR) rcs libsensor_b.a sensor_b.o

libsensor_c.a: sensor_c.o
	$(AR) rcs libsensor_c.a sensor_c.o

sensor_app: main.o libsensor_a.a libsensor_b.a libsensor_c.a
	$(CC) main.o -L. -lsensor_a -lsensor_b -lsensor_c -o sensor_app

clean:
	rm -f *.o *.a sensor_app