#include <stdio.h>
#include <stdlib.h>

int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int main(int argc, char *argv[]) {
    printf("Hello from WebAssembly!\n");
    printf("Computing Fibonacci sequence up to 10:\n");
    
    for (int i = 0; i < 10; i++) {
        int fib = fibonacci(i);
        printf("fib(%d) = %d\n", i, fib);
    }
    
    printf("\nWebAssembly module executed successfully.\n");
    
    return 0;
}