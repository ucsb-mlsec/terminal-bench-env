CC=gcc
CFLAGS=-Wall -O2
TARGET=sysinfo_static
SOURCE=sysinfo.c

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -static $(SOURCE) -o $(TARGET)

clean:
	rm -f $(TARGET)