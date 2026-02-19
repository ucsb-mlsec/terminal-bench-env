/* syscalls.c - Newlib system call stubs for bare-metal ARM Cortex-M4
 *
 * This file provides the system call interface required by newlib for
 * bare-metal embedded systems. Since there is no operating system, most
 * syscalls are stubbed out, but _sbrk() is implemented to provide heap
 * memory management.
 *
 * The heap grows upward from _heap_start to _heap_end as defined in the
 * linker script. This implementation does not support freeing memory.
 */

#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <stdint.h>

/* Linker script symbols defining heap boundaries
 * These are declared as char arrays but we only use their addresses
 */
extern char _heap_start;
extern char _heap_end;

/* Current heap pointer - tracks the end of allocated heap memory
 * Initialized to the start of the heap region
 */
static char *heap_ptr = NULL;

/* _sbrk - Increase program data space
 * 
 * This function implements dynamic memory allocation for newlib's malloc().
 * It allocates memory from the heap region defined in the linker script.
 *
 * Parameters:
 *   incr - Number of bytes to allocate
 *
 * Returns:
 *   Pointer to the start of newly allocated memory on success
 *   (void*)-1 on failure (out of memory)
 *
 * Note: This implementation does not support negative increments (freeing memory)
 */
void *_sbrk(int incr)
{
    char *prev_heap_ptr;
    
    /* Initialize heap pointer on first call */
    if (heap_ptr == NULL) {
        heap_ptr = &_heap_start;
    }
    
    /* Save current heap pointer to return */
    prev_heap_ptr = heap_ptr;
    
    /* Check if requested allocation would exceed heap bounds */
    if ((heap_ptr + incr) > &_heap_end) {
        /* Out of memory - heap would overflow into stack region */
        errno = ENOMEM;
        return (void *)-1;
    }
    
    /* Check for heap pointer underflow (negative increment beyond start) */
    if ((heap_ptr + incr) < &_heap_start) {
        errno = ENOMEM;
        return (void *)-1;
    }
    
    /* Allocation successful - update heap pointer and return old value */
    heap_ptr += incr;
    
    return (void *)prev_heap_ptr;
}

/* _write - Write to a file
 *
 * In a bare-metal system without file I/O support, this is stubbed.
 * Could be implemented to write to UART for printf() support.
 *
 * Parameters:
 *   file - File descriptor (ignored)
 *   ptr - Pointer to data buffer (ignored)
 *   len - Number of bytes to write (ignored)
 *
 * Returns:
 *   -1 (not implemented)
 */
int _write(int file, char *ptr, int len)
{
    /* Stub implementation - no file I/O available */
    (void)file;
    (void)ptr;
    (void)len;
    
    errno = ENOSYS;
    return -1;
}

/* _read - Read from a file
 *
 * In a bare-metal system without file I/O support, this is stubbed.
 * Could be implemented to read from UART.
 *
 * Parameters:
 *   file - File descriptor (ignored)
 *   ptr - Pointer to buffer (ignored)
 *   len - Number of bytes to read (ignored)
 *
 * Returns:
 *   -1 (not implemented)
 */
int _read(int file, char *ptr, int len)
{
    /* Stub implementation - no file I/O available */
    (void)file;
    (void)ptr;
    (void)len;
    
    errno = ENOSYS;
    return -1;
}

/* _close - Close a file
 *
 * In a bare-metal system without file I/O support, this is stubbed.
 *
 * Parameters:
 *   file - File descriptor (ignored)
 *
 * Returns:
 *   -1 (not implemented)
 */
int _close(int file)
{
    /* Stub implementation - no file I/O available */
    (void)file;
    
    errno = ENOSYS;
    return -1;
}

/* _lseek - Set position in a file
 *
 * In a bare-metal system without file I/O support, this is stubbed.
 *
 * Parameters:
 *   file - File descriptor (ignored)
 *   offset - File offset (ignored)
 *   whence - Reference point (ignored)
 *
 * Returns:
 *   -1 (not implemented)
 */
int _lseek(int file, int offset, int whence)
{
    /* Stub implementation - no file I/O available */
    (void)file;
    (void)offset;
    (void)whence;
    
    errno = ENOSYS;
    return -1;
}

/* _fstat - Get file status
 *
 * In a bare-metal system without file I/O support, this is stubbed.
 *
 * Parameters:
 *   file - File descriptor (ignored)
 *   st - Pointer to stat structure (ignored)
 *
 * Returns:
 *   -1 (not implemented)
 */
int _fstat(int file, struct stat *st)
{
    /* Stub implementation - no file I/O available */
    (void)file;
    (void)st;
    
    errno = ENOSYS;
    return -1;
}

/* _isatty - Check if file descriptor refers to a terminal
 *
 * In a bare-metal system, we have no terminal devices.
 *
 * Parameters:
 *   file - File descriptor (ignored)
 *
 * Returns:
 *   0 (not a terminal)
 */
int _isatty(int file)
{
    /* Stub implementation - no terminal devices available */
    (void)file;
    
    return 0;
}