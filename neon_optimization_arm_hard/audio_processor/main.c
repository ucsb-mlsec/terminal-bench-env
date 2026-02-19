#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

// External function declarations for audio processing filters
extern void apply_lowpass_filter(int16_t *samples, size_t count, float cutoff);
extern void apply_highpass_filter(int16_t *samples, size_t count, float cutoff);
extern void apply_equalizer(int16_t *samples, size_t count, float *bands, size_t band_count);
extern void apply_compression(int16_t *samples, size_t count, float threshold, float ratio);

#define SAMPLE_COUNT 1000000
#define ITERATIONS 10

int main(int argc, char **argv) {
    // Allocate memory for audio samples
    int16_t *samples = (int16_t *)malloc(SAMPLE_COUNT * sizeof(int16_t));
    if (samples == NULL) {
        fprintf(stderr, "Failed to allocate memory for samples\n");
        return 1;
    }

    // Initialize samples with test data
    printf("Initializing %d audio samples for processing...\n", SAMPLE_COUNT);
    for (size_t i = 0; i < SAMPLE_COUNT; i++) {
        // Generate simple sine-like test pattern
        samples[i] = (int16_t)((i % 1000) - 500);
    }

    // Performance benchmark - Processing audio samples
    clock_t start, end;
    double cpu_time_used;

    // Test lowpass filter
    printf("\n--- Performance benchmark: Lowpass Filter ---\n");
    start = clock();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        apply_lowpass_filter(samples, SAMPLE_COUNT, 5000.0f);
    }
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Lowpass filter: %d iterations in %.4f seconds\n", ITERATIONS, cpu_time_used);
    printf("Throughput: %.2f million samples/sec\n", 
           (SAMPLE_COUNT * ITERATIONS / 1000000.0) / cpu_time_used);

    // Test highpass filter
    printf("\n--- Performance benchmark: Highpass Filter ---\n");
    start = clock();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        apply_highpass_filter(samples, SAMPLE_COUNT, 200.0f);
    }
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Highpass filter: %d iterations in %.4f seconds\n", ITERATIONS, cpu_time_used);
    printf("Throughput: %.2f million samples/sec\n", 
           (SAMPLE_COUNT * ITERATIONS / 1000000.0) / cpu_time_used);

    // Test equalizer
    printf("\n--- Performance benchmark: Equalizer ---\n");
    float eq_bands[5] = {1.0f, 1.2f, 0.8f, 1.1f, 0.9f};
    start = clock();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        apply_equalizer(samples, SAMPLE_COUNT, eq_bands, 5);
    }
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Equalizer: %d iterations in %.4f seconds\n", ITERATIONS, cpu_time_used);
    printf("Throughput: %.2f million samples/sec\n", 
           (SAMPLE_COUNT * ITERATIONS / 1000000.0) / cpu_time_used);

    // Test compression
    printf("\n--- Performance benchmark: Compression ---\n");
    start = clock();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        apply_compression(samples, SAMPLE_COUNT, 0.7f, 3.0f);
    }
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Compression: %d iterations in %.4f seconds\n", ITERATIONS, cpu_time_used);
    printf("Throughput: %.2f million samples/sec\n", 
           (SAMPLE_COUNT * ITERATIONS / 1000000.0) / cpu_time_used);

    // Processing audio samples complete
    printf("\n=== Benchmark complete ===\n");
    printf("All audio processing filters tested successfully\n");

    // Cleanup
    free(samples);
    return 0;
}