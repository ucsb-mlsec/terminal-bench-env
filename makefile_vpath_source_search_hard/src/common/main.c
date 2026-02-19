// Main application entry point
#include "../include/memory.h"
#include "../include/utils.h"
#include "../include/processor.h"

int main(void) {
    initialize();
    
    int processor_type = get_processor_type();
    
    void* ptr = allocate_memory(100);
    
    free_memory(ptr);
    
    cleanup();
    
    return 0;
}