#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 10000

int main() {
    int *array;
    int sum = 0;
    int max_val = 0;
    int i;

    // Allocate memory for the array
    array = (int *)malloc(N * sizeof(int));
    
    // Initialize array with values
    for (i = 0; i < N; i++) {
        array[i] = i + 1;
    }

    printf("Starting GPU offload computation...\n");

    // Offload computation to GPU with RACE CONDITION
    // sum and max_val are shared but not properly reduced
    #pragma omp target map(to: array[0:N]) map(tofrom: sum, max_val)
    {
        #pragma omp parallel for shared(sum, max_val)
        for (i = 0; i < N; i++) {
            // RACE CONDITION: Multiple threads updating sum without synchronization
            sum += array[i];
            
            // RACE CONDITION: Multiple threads updating max_val without synchronization
            if (array[i] > max_val) {
                max_val = array[i];
            }
        }
    }

    printf("Sum: %d\n", sum);
    printf("Max value: %d\n", max_val);
    
    // Calculate expected values for verification
    int expected_sum = (N * (N + 1)) / 2;
    int expected_max = N;
    
    printf("Expected sum: %d\n", expected_sum);
    printf("Expected max: %d\n", expected_max);
    
    if (sum == expected_sum && max_val == expected_max) {
        printf("Results are correct (by chance - race condition exists!)\n");
    } else {
        printf("Results are incorrect due to race condition\n");
    }

    // Demonstrate correct version (commented out)
    /*
    sum = 0;
    max_val = 0;
    
    #pragma omp target map(to: array[0:N]) map(tofrom: sum, max_val)
    {
        #pragma omp parallel for reduction(+:sum) reduction(max:max_val)
        for (i = 0; i < N; i++) {
            sum += array[i];
            if (array[i] > max_val) {
                max_val = array[i];
            }
        }
    }
    printf("Corrected Sum: %d\n", sum);
    printf("Corrected Max: %d\n", max_val);
    */

    free(array);
    return 0;
}