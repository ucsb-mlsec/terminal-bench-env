#include <stdio.h>
#include <stdlib.h>

// External function declarations
void init_crypto(void);
void init_network(void);

int main(void) {
    printf("Starting netutil...\n");
    
    init_crypto();
    init_network();
    
    printf("netutil initialized successfully\n");
    
    return 0;
}