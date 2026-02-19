CC=gcc
CFLAGS=-Wall -O2
TARGET=sysmon

all: $(TARGET)

$(TARGET): sysmon.c
	$(CC) $(CFLAGS) sysmon.c -o $(TARGET)

clean:
	rm -f $(TARGET)

install: $(TARGET)
	install -m 755 $(TARGET) /usr/local/bin/

.PHONY: all clean install