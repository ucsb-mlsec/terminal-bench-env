CC = gcc
CFLAGS = -Wall -O2 -ffunction-sections -fdata-sections
LDFLAGS = -Wl,--gc-sections -Wl,-T,linker.ld

SOURCES = main.c fast_code.c slow_code.c util.c
OBJECTS = $(SOURCES:.c=.o)

TARGET = firmware.elf

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean