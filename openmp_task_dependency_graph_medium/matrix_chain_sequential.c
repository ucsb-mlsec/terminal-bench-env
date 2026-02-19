#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

/**
 * Matrix Chain Multiplication - Sequential Version
 * 
 * This program computes the minimum number of scalar multiplications
 * needed to multiply a chain of matrices using dynamic programming.
 * 
 * The algorithm uses a 2D DP table where dp[i][j] represents the
 * minimum cost to multiply matrices from position i to j.
 */

/**
 * Computes the minimum number of scalar multiplications needed
 * to multiply a chain of matrices.
 * 
 * @param p Array of dimensions where matrix i has dimensions p[i-1] x p[i]
 * @param n Number of matrices in the chain
 * @return Minimum number of scalar multiplications needed
 */
int matrixChainOrder(int *p, int n) {
    // dp[i][j] will store the minimum number of operations needed
    // to compute the matrix chain from matrix i to matrix j
    int **dp = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++) {
        dp[i] = (int *)malloc(n * sizeof(int));
        // Initialize all entries to 0
        for (int j = 0; j < n; j++) {
            dp[i][j] = 0;
        }
    }
    
    // Single matrices have zero multiplication cost
    // This is already initialized above
    
    // len is the chain length, ranging from 2 to n
    for (int len = 2; len <= n; len++) {
        // i is the starting matrix in the chain
        for (int i = 0; i < n - len + 1; i++) {
            // j is the ending matrix in the chain
            int j = i + len - 1;
            
            // Initialize to maximum value
            dp[i][j] = INT_MAX;
            
            // Try all possible split points k
            // k is the position where we split the product
            for (int k = i; k < j; k++) {
                // Cost of multiplying matrices from i to k
                // plus cost of multiplying matrices from k+1 to j
                // plus cost of multiplying the two resulting matrices
                int cost = dp[i][k] + dp[k + 1][j] + 
                          p[i] * p[k + 1] * p[j + 1];
                
                // Update minimum cost
                if (cost < dp[i][j]) {
                    dp[i][j] = cost;
                }
            }
        }
    }
    
    // The result is stored in dp[0][n-1]
    int result = dp[0][n - 1];
    
    // Free allocated memory
    for (int i = 0; i < n; i++) {
        free(dp[i]);
    }
    free(dp);
    
    return result;
}

/**
 * Creates a directory if it doesn't exist
 * 
 * @param path Directory path to create
 * @return 0 on success, -1 on failure
 */
int create_directory(const char *path) {
    struct stat st = {0};
    
    if (stat(path, &st) == -1) {
        // Directory doesn't exist, try to create it
        #ifdef _WIN32
            if (mkdir(path) != 0) {
        #else
            if (mkdir(path, 0755) != 0) {
        #endif
            if (errno != EEXIST) {
                fprintf(stderr, "Error creating directory %s: %s\n", 
                       path, strerror(errno));
                return -1;
            }
        }
    }
    return 0;
}

int main() {
    // Create output directory if it doesn't exist
    if (create_directory("/workspace/output") != 0) {
        // Try to continue anyway - the directory might exist
        // or we might have permissions to write to a non-existent path
    }
    
    // Open input file
    FILE *input_file = fopen("/workspace/input/matrix_chains.txt", "r");
    if (input_file == NULL) {
        fprintf(stderr, "Error: Could not open input file '/workspace/input/matrix_chains.txt'\n");
        fprintf(stderr, "Error details: %s\n", strerror(errno));
        return 1;
    }
    
    // Open output file
    FILE *output_file = fopen("/workspace/output/results.txt", "w");
    if (output_file == NULL) {
        fprintf(stderr, "Error: Could not open output file '/workspace/output/results.txt'\n");
        fprintf(stderr, "Error details: %s\n", strerror(errno));
        fclose(input_file);
        return 1;
    }
    
    char line[4096];
    int test_case = 0;
    
    // Process each test case
    while (fgets(line, sizeof(line), input_file) != NULL) {
        // Skip empty lines
        if (line[0] == '\n' || line[0] == '\r') {
            continue;
        }
        
        test_case++;
        
        // Parse the input line
        int num_matrices;
        int values[1000];
        int count = 0;
        
        // Read all integers from the line
        char *token = strtok(line, " \t\n\r");
        while (token != NULL && count < 1000) {
            values[count++] = atoi(token);
            token = strtok(NULL, " \t\n\r");
        }
        
        if (count < 2) {
            fprintf(stderr, "Warning: Test case %d has insufficient data\n", test_case);
            continue;
        }
        
        num_matrices = values[0];
        
        // Validate input
        if (num_matrices < 1 || count < num_matrices + 1) {
            fprintf(stderr, "Warning: Test case %d has inconsistent data\n", test_case);
            continue;
        }
        
        // Allocate array for dimensions
        // For n matrices, we need n+1 dimensions
        int *dimensions = (int *)malloc((num_matrices + 1) * sizeof(int));
        if (dimensions == NULL) {
            fprintf(stderr, "Error: Memory allocation failed for test case %d\n", test_case);
            fclose(input_file);
            fclose(output_file);
            return 1;
        }
        
        // Copy dimensions from values array
        for (int i = 0; i <= num_matrices; i++) {
            dimensions[i] = values[i + 1];
        }
        
        // Compute minimum multiplications
        int min_operations = matrixChainOrder(dimensions, num_matrices);
        
        // Write result to output file
        fprintf(output_file, "%d\n", min_operations);
        
        // Free dimensions array
        free(dimensions);
    }
    
    // Close files
    fclose(input_file);
    fclose(output_file);
    
    return 0;
}