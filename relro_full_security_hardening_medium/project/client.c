#include <stdio.h>
#include <stdlib.h>

extern void init_utils();

int main() {
    printf("Client connecting...\n");
    
    init_utils();
    
    printf("Client initialized successfully\n");
    
    return 0;
}