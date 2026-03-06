#include <stdio.h>

static int initialized = 0;

__attribute__((constructor))
static void sensor_c_init(void) {
    initialized = 1;
    printf("Sensor C constructor called\n");
}

int sensor_c_get_status(void) {
    return initialized;
}