// Generic memory implementation - can be overridden
#include "../include/memory.h"
#include <stdlib.h>

void* allocate_memory(size_t size) {
    return malloc(size);
}

void free_memory(void* ptr) {
    free(ptr);
}