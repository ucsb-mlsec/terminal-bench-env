CC=arm-none-eabi-gcc
LD=arm-none-eabi-ld
OBJDUMP=arm-none-eabi-objdump
SIZE=arm-none-eabi-size

CFLAGS=-mcpu=cortex-m4 -mthumb -O2 -g -Wall -Isrc -specs=nano.specs
LDFLAGS=-T linker_script.ld -nostartfiles --specs=nano.specs -Wl,-Map=build/output.map

SOURCES=src/main.c src/syscalls.c src/startup.c
OBJECTS=$(SOURCES:src/%.c=build/%.o)

BUILD_DIR=build

TARGET=$(BUILD_DIR)/firmware.elf
MAP=$(BUILD_DIR)/output.map
LST=$(BUILD_DIR)/firmware.lst

.PHONY: all clean size

all: $(BUILD_DIR) $(TARGET) $(LST) size

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJECTS) linker_script.ld
	$(CC) $(CFLAGS) $(LDFLAGS) $(OBJECTS) -o $@

$(LST): $(TARGET)
	$(OBJDUMP) -D $(TARGET) > $(LST)

size: $(TARGET)
	$(SIZE) $(TARGET)

clean:
	rm -rf $(BUILD_DIR)