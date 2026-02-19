#!/bin/bash
set -e

# Create the problematic main.c that will cause linker errors when modules are missing
cat > /workspace/app/main.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>

/* External function declarations for all optional modules */
void audio_init(void);
int audio_process(const char* data);
void network_init(void);
int network_process(const char* data);
void storage_init(void);
int storage_process(const char* data);

int main(void) {
    printf("Starting application...\n");
    
    /* Initialize all modules */
    audio_init();
    network_init();
    storage_init();
    
    /* Process data with all modules */
    int audio_result = audio_process("test audio");
    int network_result = network_process("test network");
    int storage_result = storage_process("test storage");
    
    printf("Audio processing result: %d\n", audio_result);
    printf("Network processing result: %d\n", network_result);
    printf("Storage processing result: %d\n", storage_result);
    
    printf("Application completed successfully\n");
    return 0;
}
EOF

# Create audio.c module
cat > /workspace/app/audio.c << 'EOF'
#include <stdio.h>

void audio_init(void) {
    printf("Audio module initialized\n");
}

int audio_process(const char* data) {
    printf("Audio processing: %s\n", data);
    return 1;
}
EOF

# Create network.c module
cat > /workspace/app/network.c << 'EOF'
#include <stdio.h>

void network_init(void) {
    printf("Network module initialized\n");
}

int network_process(const char* data) {
    printf("Network processing: %s\n", data);
    return 2;
}
EOF

# Create storage.c module
cat > /workspace/app/storage.c << 'EOF'
#include <stdio.h>

void storage_init(void) {
    printf("Storage module initialized\n");
}

int storage_process(const char* data) {
    printf("Storage processing: %s\n", data);
    return 3;
}
EOF

# Create original Makefile that will fail with missing modules
cat > /workspace/app/Makefile << 'EOF'
CC = gcc
CFLAGS = -Wall -Wextra

SOURCES = main.c
MODULES = 

ifdef WITH_AUDIO
    MODULES += audio.c
endif

ifdef WITH_NETWORK
    MODULES += network.c
endif

ifdef WITH_STORAGE
    MODULES += storage.c
endif

all: app

app: $(SOURCES) $(MODULES)
	$(CC) $(CFLAGS) -o app $(SOURCES) $(MODULES)

clean:
	rm -f app *.o

.PHONY: all clean
EOF

# Now fix the problem by implementing weak linking approach

# Create a new main.c with weak symbol declarations
cat > /workspace/app/main.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>

/* Weak symbol declarations - these allow linking even when modules are missing */
void audio_init(void) __attribute__((weak));
int audio_process(const char* data) __attribute__((weak));
void network_init(void) __attribute__((weak));
int network_process(const char* data) __attribute__((weak));
void storage_init(void) __attribute__((weak));
int storage_process(const char* data) __attribute__((weak));

/* Wrapper functions that check if the real implementation exists */
static void safe_audio_init(void) {
    if (audio_init) {
        audio_init();
    }
}

static int safe_audio_process(const char* data) {
    if (audio_process) {
        return audio_process(data);
    }
    return 0;
}

static void safe_network_init(void) {
    if (network_init) {
        network_init();
    }
}

static int safe_network_process(const char* data) {
    if (network_process) {
        return network_process(data);
    }
    return 0;
}

static void safe_storage_init(void) {
    if (storage_init) {
        storage_init();
    }
}

static int safe_storage_process(const char* data) {
    if (storage_process) {
        return storage_process(data);
    }
    return 0;
}

int main(void) {
    printf("Starting application...\n");
    
    /* Initialize all modules (safely) */
    safe_audio_init();
    safe_network_init();
    safe_storage_init();
    
    /* Process data with all modules (safely) */
    int audio_result = safe_audio_process("test audio");
    int network_result = safe_network_process("test network");
    int storage_result = safe_storage_process("test storage");
    
    printf("Audio processing result: %d\n", audio_result);
    printf("Network processing result: %d\n", network_result);
    printf("Storage processing result: %d\n", storage_result);
    
    printf("Application completed successfully\n");
    return 0;
}
EOF

# Update Makefile to support flexible module selection
cat > /workspace/app/Makefile << 'EOF'
CC = gcc
CFLAGS = -Wall -Wextra
LDFLAGS = 

SOURCES = main.c
MODULES = 

ifdef WITH_AUDIO
    MODULES += audio.c
    CFLAGS += -DHAVE_AUDIO
endif

ifdef WITH_NETWORK
    MODULES += network.c
    CFLAGS += -DHAVE_NETWORK
endif

ifdef WITH_STORAGE
    MODULES += storage.c
    CFLAGS += -DHAVE_STORAGE
endif

all: app

app: $(SOURCES) $(MODULES)
	$(CC) $(CFLAGS) -o app $(SOURCES) $(MODULES) $(LDFLAGS)

clean:
	rm -f app *.o

.PHONY: all clean
EOF

cd /workspace/app

# Test Scenario 1: Build with all three modules present
echo "=== Test Scenario 1: Building with all modules ==="
make clean
make WITH_AUDIO=1 WITH_NETWORK=1 WITH_STORAGE=1
if [ $? -ne 0 ]; then
    echo "FAIL: Build with all modules failed"
    exit 1
fi
./app
if [ $? -ne 0 ]; then
    echo "FAIL: Execution with all modules failed"
    exit 1
fi
echo "SUCCESS: All modules test passed"
echo ""

# Test Scenario 2: Build with only audio and network modules
echo "=== Test Scenario 2: Building with audio and network only ==="
make clean
make WITH_AUDIO=1 WITH_NETWORK=1
if [ $? -ne 0 ]; then
    echo "FAIL: Build with audio and network failed"
    exit 1
fi
./app
if [ $? -ne 0 ]; then
    echo "FAIL: Execution with audio and network failed"
    exit 1
fi
echo "SUCCESS: Audio and network test passed"
echo ""

# Test Scenario 3: Build with only storage module
echo "=== Test Scenario 3: Building with storage only ==="
make clean
make WITH_STORAGE=1
if [ $? -ne 0 ]; then
    echo "FAIL: Build with storage only failed"
    exit 1
fi
./app
if [ $? -ne 0 ]; then
    echo "FAIL: Execution with storage only failed"
    exit 1
fi
echo "SUCCESS: Storage only test passed"
echo ""

# Test Scenario 4: Build with no optional modules
echo "=== Test Scenario 4: Building with no modules ==="
make clean
make
if [ $? -ne 0 ]; then
    echo "FAIL: Build with no modules failed"
    exit 1
fi
./app
if [ $? -ne 0 ]; then
    echo "FAIL: Execution with no modules failed"
    exit 1
fi
echo "SUCCESS: No modules test passed"
echo ""

# Final cleanup
make clean

echo "=== ALL TESTS PASSED ==="
exit 0