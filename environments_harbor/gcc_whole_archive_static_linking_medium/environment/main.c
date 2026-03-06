#include <stdio.h>
#include <stdlib.h>

typedef struct {
    char* name;
    char* version;
} Plugin;

Plugin plugins[10];
int plugin_count = 0;

void register_plugin(char* name, char* version) {
    if (plugin_count < 10) {
        plugins[plugin_count].name = name;
        plugins[plugin_count].version = version;
        plugin_count++;
    }
}

int main() {
    printf("Plugin System Starting...\n");
    
    if (plugin_count == 0) {
        printf("No plugins loaded\n");
    } else {
        printf("Loaded %d plugin(s):\n", plugin_count);
        for (int i = 0; i < plugin_count; i++) {
            printf("  - %s (version %s)\n", plugins[i].name, plugins[i].version);
        }
    }
    
    return 0;
}