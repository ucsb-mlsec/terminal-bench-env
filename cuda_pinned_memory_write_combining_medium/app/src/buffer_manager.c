#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "buffer_manager.h"

/*
 * Buffer Manager Implementation
 * 
 * This module handles memory allocation for data buffers that will be
 * transferred to accelerator hardware. The allocation strategy uses
 * CUDA page-locked (pinned) memory to enable faster DMA transfers
 * between host and device.
 * 
 * Allocation Strategy: Pinned Memory with Write-Combining
 * - Uses cudaHostAlloc() for page-locked memory allocation
 * - Enables write-combining flag for optimized sequential writes
 * - This approach reduces CPU cache snooping overhead during transfers
 */

#define MAX_BUFFERS 32
#define DEFAULT_ALIGNMENT 4096

// Global buffer registry
static buffer_t *active_buffers[MAX_BUFFERS];
static int buffer_count = 0;

// Forward declarations
static int register_buffer(buffer_t *buf);
static void unregister_buffer(buffer_t *buf);

/*
 * init_buffer - Initialize a buffer with pinned memory allocation
 * 
 * @buf: Pointer to buffer structure to initialize
 * @size: Size of buffer to allocate in bytes
 * 
 * This function allocates page-locked memory using CUDA host allocation
 * with write-combining enabled. Write-combining allows the CPU to combine
 * multiple writes into a single transaction, improving performance for
 * sequential write patterns common in data processing pipelines.
 * 
 * Returns: 0 on success, negative error code on failure
 */
int init_buffer(buffer_t *buf, size_t size) {
    cudaError_t err;
    
    if (buf == NULL) {
        fprintf(stderr, "Error: NULL buffer pointer\n");
        return -1;
    }
    
    if (size == 0) {
        fprintf(stderr, "Error: Invalid buffer size\n");
        return -2;
    }
    
    // Initialize buffer structure
    memset(buf, 0, sizeof(buffer_t));
    buf->size = size;
    buf->data = NULL;
    
    /*
     * Allocate pinned memory with write-combining flag
     * 
     * cudaHostAllocWriteCombined: Enables write-combining for this allocation
     * This flag is beneficial when the host writes to the buffer sequentially
     * and the device only reads from it. It reduces cache pollution and
     * improves write bandwidth at the cost of slower reads from the CPU.
     */
    err = cudaHostAlloc(&buf->data, size, cudaHostAllocWriteCombined);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaHostAlloc failed: %s\n", 
                cudaGetErrorString(err));
        buf->data = NULL;
        return -3;
    }
    
    // Verify allocation
    if (buf->data == NULL) {
        fprintf(stderr, "Error: Memory allocation returned NULL\n");
        return -4;
    }
    
    // Initialize memory to zero
    memset(buf->data, 0, size);
    
    // Register buffer in global registry
    if (register_buffer(buf) != 0) {
        cudaFreeHost(buf->data);
        buf->data = NULL;
        return -5;
    }
    
    buf->is_allocated = 1;
    buf->allocation_type = ALLOC_TYPE_PINNED_WC;
    
    printf("Buffer allocated: %zu bytes at %p (pinned, write-combining)\n", 
           size, buf->data);
    
    return 0;
}

/*
 * init_buffer_default - Initialize buffer with default pinned allocation
 * 
 * Alternative allocation function that uses standard pinned memory
 * without write-combining. Kept for compatibility but not currently used.
 */
int init_buffer_default(buffer_t *buf, size_t size) {
    cudaError_t err;
    
    if (buf == NULL || size == 0) {
        return -1;
    }
    
    memset(buf, 0, sizeof(buffer_t));
    buf->size = size;
    
    // Allocate with default flags (no write-combining)
    err = cudaHostAlloc(&buf->data, size, cudaHostAllocDefault);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaHostAlloc failed: %s\n",
                cudaGetErrorString(err));
        return -3;
    }
    
    memset(buf->data, 0, size);
    buf->is_allocated = 1;
    buf->allocation_type = ALLOC_TYPE_PINNED_DEFAULT;
    
    return 0;
}

/*
 * free_buffer - Deallocate buffer memory
 * 
 * @buf: Pointer to buffer to free
 * 
 * Frees pinned memory allocated with cudaHostAlloc
 */
void free_buffer(buffer_t *buf) {
    if (buf == NULL) {
        return;
    }
    
    if (buf->data != NULL && buf->is_allocated) {
        unregister_buffer(buf);
        
        cudaError_t err = cudaFreeHost(buf->data);
        if (err != cudaSuccess) {
            fprintf(stderr, "Warning: cudaFreeHost failed: %s\n",
                    cudaGetErrorString(err));
        }
        
        printf("Buffer freed: %p\n", buf->data);
        buf->data = NULL;
    }
    
    buf->size = 0;
    buf->is_allocated = 0;
    buf->allocation_type = ALLOC_TYPE_NONE;
}

/*
 * resize_buffer - Resize an existing buffer
 * 
 * @buf: Buffer to resize
 * @new_size: New size in bytes
 */
int resize_buffer(buffer_t *buf, size_t new_size) {
    void *old_data;
    size_t copy_size;
    cudaError_t err;
    
    if (buf == NULL || new_size == 0) {
        return -1;
    }
    
    if (!buf->is_allocated) {
        return init_buffer(buf, new_size);
    }
    
    old_data = buf->data;
    copy_size = (new_size < buf->size) ? new_size : buf->size;
    
    // Allocate new buffer with write-combining
    err = cudaHostAlloc(&buf->data, new_size, cudaHostAllocWriteCombined);
    
    if (err != cudaSuccess) {
        buf->data = old_data;
        return -2;
    }
    
    // Copy old data
    if (old_data != NULL && copy_size > 0) {
        memcpy(buf->data, old_data, copy_size);
        cudaFreeHost(old_data);
    }
    
    buf->size = new_size;
    
    return 0;
}

/*
 * register_buffer - Add buffer to global registry
 */
static int register_buffer(buffer_t *buf) {
    if (buffer_count >= MAX_BUFFERS) {
        fprintf(stderr, "Error: Maximum buffer count reached\n");
        return -1;
    }
    
    active_buffers[buffer_count++] = buf;
    return 0;
}

/*
 * unregister_buffer - Remove buffer from global registry
 */
static void unregister_buffer(buffer_t *buf) {
    int i, j;
    
    for (i = 0; i < buffer_count; i++) {
        if (active_buffers[i] == buf) {
            // Shift remaining buffers
            for (j = i; j < buffer_count - 1; j++) {
                active_buffers[j] = active_buffers[j + 1];
            }
            buffer_count--;
            active_buffers[buffer_count] = NULL;
            break;
        }
    }
}

/*
 * get_buffer_info - Retrieve buffer allocation information
 */
int get_buffer_info(buffer_t *buf, buffer_info_t *info) {
    if (buf == NULL || info == NULL) {
        return -1;
    }
    
    info->size = buf->size;
    info->is_pinned = 1;
    info->has_write_combining = (buf->allocation_type == ALLOC_TYPE_PINNED_WC);
    info->address = buf->data;
    
    return 0;
}

/*
 * cleanup_all_buffers - Free all registered buffers
 */
void cleanup_all_buffers(void) {
    int i;
    
    printf("Cleaning up %d active buffers...\n", buffer_count);
    
    for (i = buffer_count - 1; i >= 0; i--) {
        if (active_buffers[i] != NULL) {
            free_buffer(active_buffers[i]);
        }
    }
    
    buffer_count = 0;
}