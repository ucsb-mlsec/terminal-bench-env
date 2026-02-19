JAVA_HOME = /usr/lib/jvm/default-java
CC = gcc
CFLAGS = -fPIC -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux

TARGET = libstring.so

all: $(TARGET)

$(TARGET): StringManipulator.c
	$(CC) $(CFLAGS) -shared -o $(TARGET) StringManipulator.c

clean:
	rm -f $(TARGET)