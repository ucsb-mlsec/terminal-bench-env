#ifndef MEMORY_H
#define MEMORY_H

#include <stddef.h>

void* allocate_memory(size_t size);
void free_memory(void* ptr);

#endif