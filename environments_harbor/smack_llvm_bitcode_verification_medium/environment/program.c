#include <assert.h>
#include <stdio.h>

int main() {
    int n = 10;
    int sum = 0;
    int product = 1;
    
    // Compute sum and product of integers from 1 to n
    for (int i = 1; i <= n; i++) {
        sum += i;
        product *= i;
    }
    
    // Assert that sum equals n*(n+1)/2
    assert(sum == (n * (n + 1)) / 2);
    
    // Assert that product is positive
    assert(product > 0);
    
    // Array bounds checking scenario
    int arr[5] = {10, 20, 30, 40, 50};
    int index = 0;
    
    for (int i = 0; i < 8; i++) {
        if (i < 5) {
            index = i;
        }
    }
    
    // Assert that index is within valid array bounds
    assert(index >= 0 && index < 5);
    
    // Mathematical invariant check
    int x = 15;
    int y = 7;
    int z = x - y;
    
    // Assert mathematical property
    assert(z == 8);
    
    // State machine simulation
    int state = 0;
    for (int step = 0; step < 5; step++) {
        if (state == 0) {
            state = 1;
        } else if (state == 1) {
            state = 2;
        } else {
            state = 3;
        }
    }
    
    // Assert that state is in valid range
    assert(state >= 0 && state <= 3);
    
    printf("All assertions passed!\n");
    
    return 0;
}