#include <stdio.h>
#include <stdlib.h>

// External function declaration
void print_message(void);

int main(void) {
    // Print version information
    printf("CLI Tool v1.0\n");
    
    // Call external function
    print_message();
    
    // Print completion message
    printf("Execution complete\n");
    
    return 0;
}