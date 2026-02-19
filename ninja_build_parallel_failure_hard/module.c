#include <stdio.h>
#include "types.h"
#include "config.h"

void module_init(void) {
    printf("Module initialized with version %d\n", CONFIG_VERSION);
}

int module_process(DataType *data) {
    if (data == NULL) return -1;
    data->value = CONFIG_DEFAULT_VALUE;
    return 0;
}