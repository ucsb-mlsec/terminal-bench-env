CC = gcc
WINDOWS_CC = x86_64-w64-mingw32-gcc
CFLAGS = -O2 -Wall
WINDOWS_CFLAGS = -O2 -Wall

all: linux windows

linux: compressor.c
	$(CC) $(CFLAGS) compressor.c -o compressor

windows: compressor.c
	mkdir -p build
	$(WINDOWS_CC) $(WINDOWS_CFLAGS) compressor.c -o build/compressor.exe

clean:
	rm -f compressor
	rm -f build/compressor.exe
	rmdir build 2>/dev/null || true

.PHONY: all linux windows clean