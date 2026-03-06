CC = gcc
CXX = g++
AS = nasm
LD = ld

CFLAGS = -m32 -c -O2 -Wall
LDFLAGS = -m elf_i386
ASFLAGS = -f elf32

SOURCES = bootloader.c
ASM_SOURCES = boot.asm
OBJECTS = bootloader.o boot.o

all: boot.bin

bootloader.o: bootloader.c
	$(CC) $(CFLAGS) bootloader.c -o bootloader.o

boot.o: boot.asm
	$(AS) $(ASFLAGS) boot.asm -o boot.o

boot.bin: $(OBJECTS)
	$(LD) $(LDFLAGS) -o boot.bin $(OBJECTS)
	objcopy -O binary boot.bin boot.bin

clean:
	rm -f *.o boot.bin boot.elf

.PHONY: all clean