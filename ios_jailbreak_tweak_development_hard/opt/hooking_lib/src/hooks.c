#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

#ifndef RTLD_NEXT
#define RTLD_NEXT ((void *) -1l)
#endif

// Function pointer types for original functions
typedef char* (*original_strcpy_type)(char *dest, const char *src);
typedef void* (*original_malloc_type)(size_t size);
typedef void (*original_free_type)(void *ptr);
typedef int (*original_printf_type)(const char *format, ...);

// Global counters for statistics
static unsigned long malloc_count = 0;
static unsigned long free_count = 0;
static unsigned long strcpy_count = 0;

/*
 * Hook implementation for strcpy
 * Intercepts strcpy calls and logs them for debugging
 * This is a proof-of-concept implementation
 */
char* strcpy(char *dest, const char *src) {
    original_strcpy_type original_strcpy;
    char buffer[64];  // Local buffer for logging/processing
    
    // Get pointer to original strcpy function
    original_strcpy = (original_strcpy_type)dlsym(RTLD_NEXT, "strcpy");
    
    if (original_strcpy == NULL) {
        fprintf(stderr, "Error: Could not find original strcpy\n");
        return dest;
    }
    
    // Increment counter
    strcpy_count++;
    
    // Copy source to local buffer for logging purposes
    // This allows us to inspect/log the data being copied
    original_strcpy(buffer, src);
    
    // Log the operation (uncomment for debugging)
    // fprintf(stderr, "[HOOK] strcpy called (count: %lu): %s\n", strcpy_count, buffer);
    
    // Perform the actual copy to destination
    return original_strcpy(dest, src);
}

/*
 * Hook implementation for malloc
 * Tracks memory allocations for debugging
 */
void* malloc(size_t size) {
    original_malloc_type original_malloc;
    void *ptr;
    
    // Get pointer to original malloc function
    original_malloc = (original_malloc_type)dlsym(RTLD_NEXT, "malloc");
    
    if (original_malloc == NULL) {
        fprintf(stderr, "Error: Could not find original malloc\n");
        return NULL;
    }
    
    // Call original malloc
    ptr = original_malloc(size);
    
    // Track allocation
    if (ptr != NULL) {
        malloc_count++;
        // fprintf(stderr, "[HOOK] malloc(%zu) = %p (count: %lu)\n", size, ptr, malloc_count);
    }
    
    return ptr;
}

/*
 * Hook implementation for free
 * Tracks memory deallocations for debugging
 */
void free(void *ptr) {
    original_free_type original_free;
    
    // Get pointer to original free function
    original_free = (original_free_type)dlsym(RTLD_NEXT, "free");
    
    if (original_free == NULL) {
        fprintf(stderr, "Error: Could not find original free\n");
        return;
    }
    
    // Track deallocation
    if (ptr != NULL) {
        free_count++;
        // fprintf(stderr, "[HOOK] free(%p) (count: %lu)\n", ptr, free_count);
    }
    
    // Call original free
    original_free(ptr);
}

/*
 * Hook implementation for printf
 * Intercepts printf calls for logging
 */
int printf(const char *format, ...) {
    original_printf_type original_printf;
    va_list args;
    int result;
    
    // Get pointer to original printf function
    original_printf = (original_printf_type)dlsym(RTLD_NEXT, "printf");
    
    if (original_printf == NULL) {
        fprintf(stderr, "Error: Could not find original printf\n");
        return -1;
    }
    
    // Forward to original printf with variable arguments
    va_start(args, format);
    result = vprintf(format, args);
    va_end(args);
    
    return result;
}

/*
 * Library initialization function
 * Called when the shared library is loaded
 */
__attribute__((constructor))
static void hook_init(void) {
    fprintf(stderr, "[HOOK] Function hooking library loaded\n");
    fprintf(stderr, "[HOOK] Intercepting: strcpy, malloc, free, printf\n");
}

/*
 * Library cleanup function
 * Called when the shared library is unloaded
 */
__attribute__((destructor))
static void hook_fini(void) {
    fprintf(stderr, "[HOOK] Function hooking library unloaded\n");
    fprintf(stderr, "[HOOK] Statistics:\n");
    fprintf(stderr, "[HOOK]   strcpy calls: %lu\n", strcpy_count);
    fprintf(stderr, "[HOOK]   malloc calls: %lu\n", malloc_count);
    fprintf(stderr, "[HOOK]   free calls: %lu\n", free_count);
}