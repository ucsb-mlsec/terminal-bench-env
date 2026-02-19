// Plugin 2 - Video Plugin
// This plugin automatically registers itself at startup using a constructor function

#include <stdio.h>

// External registration function provided by the main application
extern void register_plugin(const char* name, const char* version);

// Constructor function that runs automatically at program startup
// The __attribute__((constructor)) tells the compiler to execute this
// function before main() is called
void __attribute__((constructor)) plugin2_init(void) {
    register_plugin("VideoPlugin", "2.1");
}