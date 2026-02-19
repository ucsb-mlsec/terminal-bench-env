#ifndef HASH_H
#define HASH_H

#include <stddef.h>

void sha256_init(void* context);
void sha256_hash(void* context, const char* input, char* output);

#endif