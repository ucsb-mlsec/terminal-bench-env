#include <stdio.h>
#include "utils/helper.h"
#include "core/processor.h"

int main() {
    printf("Starting application...\n");
    
    initialize_utils();
    process_data();
    
    printf("Application completed successfully.\n");
    return 0;
}