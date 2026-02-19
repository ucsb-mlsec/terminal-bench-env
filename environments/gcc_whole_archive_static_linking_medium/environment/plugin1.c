#include <stdio.h>

// Plugin 1 - Audio Plugin

// External registration function provided by main application
extern void register_plugin(const char* name, const char* version);

// Constructor function that runs automatically at startup
static void __attribute__((constructor)) plugin1_init(void) {
    register_plugin("AudioPlugin", "1.0");
}