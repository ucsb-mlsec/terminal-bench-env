#include <stdio.h>

static int initialized = 0;

__attribute__((constructor))
static void sensor_b_init(void) {
    initialized = 1;
    printf("Sensor B constructor called\n");
}

int sensor_b_get_status(void) {
    return initialized;
}