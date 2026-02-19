#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <string.h>

/*
 * Hook Loader - LD_PRELOAD Library Initializer
 * 
 * This library demonstrates iOS-style method hooking using LD_PRELOAD
 * on Linux systems. When loaded via LD_PRELOAD, this library intercepts
 * standard C library function calls.
 *
 * Usage: LD_PRELOAD=/opt/hooking_lib/libhooks.so ./target_program
 */

// Global state for tracking hook activity
static int hooks_initialized = 0;
static FILE *hook_log = NULL;

/*
 * Library constructor - executed when library is loaded
 * Runs before main() of the target program
 */
__attribute__((constructor))
static void init_hooks(void) {
    // Open log file for hook activity
    hook_log = fopen("/var/log/hooks.log", "a");
    
    fprintf(stderr, "[HOOK_LOADER] Hooking library loaded via LD_PRELOAD\n");
    fprintf(stderr, "[HOOK_LOADER] Initializing function interception layer\n");
    
    if (hook_log) {
        fprintf(hook_log, "=== Hook library initialized ===\n");
        fflush(hook_log);
    }
    
    hooks_initialized = 1;
    
    fprintf(stderr, "[HOOK_LOADER] All hooks are now active\n");
}

/*
 * Library destructor - executed when library is unloaded
 * Runs after main() completes or program exits
 */
__attribute__((destructor))
static void cleanup_hooks(void) {
    fprintf(stderr, "[HOOK_LOADER] Cleaning up hooking library\n");
    
    if (hook_log) {
        fprintf(hook_log, "=== Hook library unloaded ===\n");
        fclose(hook_log);
    }
    
    hooks_initialized = 0;
}