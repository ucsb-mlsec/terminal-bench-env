#include <stdio.h>
#include <stdlib.h>
#include "config.h"
#include "generated_defs.h"

void print_config_info(void) {
    printf("Application: %s\n", APP_NAME);
    printf("Version: %s\n", APP_VERSION);
    printf("Build number: %d\n", BUILD_NUMBER);
}

int get_max_buffer_size(void) {
    return MAX_BUFFER_SIZE * 2;
}