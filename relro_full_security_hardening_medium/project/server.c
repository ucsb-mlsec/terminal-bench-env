#include <stdio.h>
#include <stdlib.h>

extern void init_utils();

int main(int argc, char *argv[]) {
    printf("Server starting...\n");
    
    // Initialize utilities library
    init_utils();
    
    printf("Server initialized successfully\n");
    printf("Listening for connections...\n");
    
    // Main server loop would go here
    // For now, just exit successfully
    
    printf("Server shutting down\n");
    
    return 0;
}