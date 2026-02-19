// Plugin 3: Network Plugin
#include <stdio.h>

// External registration function
extern void register_plugin(const char* name, const char* version);

// Plugin initialization function that runs automatically at startup
void __attribute__((constructor)) plugin3_init(void) {
    register_plugin("NetworkPlugin", "1.5");
}