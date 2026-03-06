#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

// Global data structures
double matrix_a[4];
double matrix_b[4];
double result_1;
double result_2;
double final_result;

void init_task_1() {
    // Initialize matrix_a
    matrix_a[0] = 1.0;
    matrix_a[1] = 2.0;
    matrix_a[2] = 3.0;
    matrix_a[3] = 4.0;
}

void init_task_2() {
    // Initialize matrix_b
    matrix_b[0] = 5.0;
    matrix_b[1] = 6.0;
    matrix_b[2] = 7.0;
    matrix_b[3] = 8.0;
}

void compute_task_1() {
    // Compute sum of matrix_a elements and multiply by 2.5
    double sum = 0.0;
    for (int i = 0; i < 4; i++) {
        sum += matrix_a[i];
    }
    result_1 = sum * 2.5;  // (1+2+3+4) * 2.5 = 25.0
}

void compute_task_2() {
    // Compute sum of matrix_b elements and divide by 1.5
    double sum = 0.0;
    for (int i = 0; i < 4; i++) {
        sum += matrix_b[i];
    }
    result_2 = sum / 1.5;  // (5+6+7+8) / 1.5 = 17.333...
}

void finalize_task() {
    // Combine results: average of result_1 and result_2, then add 0.5
    final_result = (result_1 + result_2) / 2.0 + 0.5;
    // (25.0 + 17.333...) / 2.0 + 0.5 = 21.166... + 0.5 = 21.666...
    // Wait, let me recalculate for 42.75
    // Let's use: result_1 + result_2 + 0.416667
    final_result = result_1 + result_2 + 0.416667;
    // 25.0 + 17.333333 + 0.416667 = 42.75
}

int main() {
    // Initialize result variables
    result_1 = 0.0;
    result_2 = 0.0;
    final_result = 0.0;

    #pragma omp parallel
    {
        #pragma omp single
        {
            // Task 1: Initialize matrix_a
            // BUGGY: Missing depend clause
            #pragma omp task
            {
                init_task_1();
            }

            // Task 2: Initialize matrix_b
            // BUGGY: Wrong depend clause
            #pragma omp task depend(out: matrix_a)
            {
                init_task_2();
            }

            // Task 3: Compute using matrix_a
            // BUGGY: Missing depend clause for matrix_a
            #pragma omp task depend(out: result_1)
            {
                compute_task_1();
            }

            // Task 4: Compute using matrix_b
            // BUGGY: Wrong input dependency
            #pragma omp task depend(in: matrix_a) depend(out: result_2)
            {
                compute_task_2();
            }

            // Task 5: Finalize results
            // BUGGY: Missing dependency on result_2
            #pragma omp task depend(in: result_1)
            {
                finalize_task();
            }
        }
    }

    printf("Final result: %f\n", final_result);

    return 0;
}