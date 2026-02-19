CC=gcc
CFLAGS=-Wall -O2 -m32
INCLUDES=-I$(JAVA_HOME)/include
TARGET=libsensor.so
SOURCES=sensor.c
OBJECTS=sensor.o

all: $(TARGET)

$(OBJECTS): $(SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -c $(SOURCES) -o $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -shared -o $(TARGET) $(OBJECTS)

clean:
	rm -f $(OBJECTS) $(TARGET)