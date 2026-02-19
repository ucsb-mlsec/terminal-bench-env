#include <stdio.h>
#include "config.h"
#include "types.h"

int main(void) {
    status_t result = STATUS_OK;
    
    printf("Application: %s\n", APP_NAME);
    printf("Version: %d.%d\n", VERSION_MAJOR, VERSION_MINOR);
    printf("Status: %d\n", result);
    
    return 0;
}