#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

// Slow loop-based checksum computation
// Computes parity by iterating through each bit position
uint32_t compute_checksum(uint32_t value) {
    uint32_t checksum = 0;
    
    // Iterate through each bit position (0-31)
    for (int bit = 0; bit < 32; bit++) {
        // Check if bit is set using shift and mask
        if ((value >> bit) & 1) {
            // XOR accumulation for parity-based checksum
            checksum ^= (bit + 1);
        }
    }
    
    // Additional loop to make it even slower
    // Count total set bits
    int bit_count = 0;
    for (int bit = 0; bit < 32; bit++) {
        if ((value >> bit) & 1) {
            bit_count++;
        }
    }
    
    // Incorporate bit count into checksum
    checksum ^= bit_count;
    checksum ^= value;
    
    return checksum;
}

int main() {
    FILE *fp = fopen("transactions.dat", "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open transactions.dat\n");
        return 1;
    }
    
    // Determine file size to calculate number of transactions
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    size_t num_transactions = file_size / sizeof(uint32_t);
    
    // Allocate memory for transaction IDs
    uint32_t *transactions = (uint32_t*)malloc(num_transactions * sizeof(uint32_t));
    if (!transactions) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(fp);
        return 1;
    }
    
    // Read transaction IDs from file
    size_t read_count = fread(transactions, sizeof(uint32_t), num_transactions, fp);
    fclose(fp);
    
    if (read_count != num_transactions) {
        fprintf(stderr, "Error: Failed to read all transactions\n");
        free(transactions);
        return 1;
    }
    
    // Record start time
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Process all transactions and accumulate checksum
    uint64_t final_checksum = 0;
    for (size_t i = 0; i < num_transactions; i++) {
        uint32_t cs = compute_checksum(transactions[i]);
        final_checksum += cs;
    }
    
    // Record end time
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    // Calculate elapsed time in milliseconds
    long seconds = end.tv_sec - start.tv_sec;
    long nanoseconds = end.tv_nsec - start.tv_nsec;
    long elapsed_ms = seconds * 1000 + nanoseconds / 1000000;
    
    // Print results
    printf("Execution time: %ld ms\n", elapsed_ms);
    printf("Final checksum: %llu\n", (unsigned long long)final_checksum);
    
    free(transactions);
    return 0;
}