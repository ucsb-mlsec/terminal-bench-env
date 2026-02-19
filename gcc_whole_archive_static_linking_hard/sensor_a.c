#include <stdio.h>

static int initialized = 0;

static void __attribute__((constructor)) sensor_a_init(void)
{
    initialized = 1;
    printf("Sensor A constructor called\n");
}

int sensor_a_get_status(void)
{
    return initialized;
}