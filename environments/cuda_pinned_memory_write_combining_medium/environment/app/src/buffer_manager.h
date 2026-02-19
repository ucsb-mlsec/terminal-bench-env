#ifndef BUFFER_MANAGER_H
#define BUFFER_MANAGER_H

#include <stddef.h>

/*
 * Buffer Manager Header
 * 
 * This module provides memory buffer management for high-performance
 * data transfers to accelerator hardware. Buffers are allocated with
 * specific memory characteristics to optimize transfer rates.
 */

/* Buffer allocation flags */
#define BUFFER_FLAG_NONE        0x00
#define BUFFER_FLAG_ALLOCATED   0x01
#define BUFFER_FLAG_PINNED      0x02
#define BUFFER_FLAG_WC          0x04  /* Write-combining */

/* Default buffer sizes */
#define DEFAULT_BUFFER_SIZE     (4 * 1024 * 1024)  /* 4 MB */
#define MAX_BUFFER_SIZE         (256 * 1024 * 1024) /* 256 MB */
#define MIN_BUFFER_SIZE         (64 * 1024)         /* 64 KB */

/*
 * buffer_t - Data buffer structure
 * 
 * @data: Pointer to allocated memory region
 * @size: Size of the buffer in bytes
 * @flags: Status and allocation flags
 * 
 * This structure represents a memory buffer that can be used for
 * data transfers to hardware accelerators. The allocation method
 * and memory properties are tracked via the flags field.
 */
typedef struct {
    void *data;
    size_t size;
    int flags;
} buffer_t;

/*
 * init_buffer - Initialize and allocate a buffer
 * @buf: Pointer to buffer structure to initialize
 * @size: Size in bytes to allocate
 * 
 * Allocates memory for the buffer using optimized allocation
 * methods suitable for hardware transfers.
 * 
 * Return: 0 on success, negative error code on failure
 */
int init_buffer(buffer_t *buf, size_t size);

/*
 * free_buffer - Free buffer resources
 * @buf: Pointer to buffer structure to free
 * 
 * Releases all memory associated with the buffer and resets
 * the structure fields.
 */
void free_buffer(buffer_t *buf);

/*
 * reset_buffer - Reset buffer contents
 * @buf: Pointer to buffer structure to reset
 * 
 * Clears the buffer contents without deallocating memory.
 * 
 * Return: 0 on success, negative error code on failure
 */
int reset_buffer(buffer_t *buf);

#endif /* BUFFER_MANAGER_H */