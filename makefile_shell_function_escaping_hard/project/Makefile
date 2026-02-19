CC=gcc
CFLAGS=-Wall

OUTPUT_DIR=$(shell cat config.txt | grep OUTPUT | cut -d= -f2)
DYNAMIC_DIR=$(shell echo 'build output')

.PHONY: all clean

all: setup compile

setup:
	mkdir -p $(OUTPUT_DIR)
	mkdir -p $(DYNAMIC_DIR)

compile: setup
	$(CC) $(CFLAGS) src/main.c -o $(OUTPUT_DIR)/hello
	$(CC) $(CFLAGS) -c src/main.c -o $(DYNAMIC_DIR)/main.o

clean:
	rm -rf $(OUTPUT_DIR)/*.o
	rm -rf $(DYNAMIC_DIR)/*.o
	rm -f $(OUTPUT_DIR)/hello