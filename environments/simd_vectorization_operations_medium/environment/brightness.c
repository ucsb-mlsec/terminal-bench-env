#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

// Naive brightness adjustment - processes one pixel at a time
// This is intentionally slow and unoptimized
void adjust_brightness_naive(uint8_t* image, int width, int height, int brightness_delta) {
    // Process each row
    for (int y = 0; y < height; y++) {
        // Process each column
        for (int x = 0; x < width; x++) {
            int index = y * width + x;
            int pixel_value = image[index];
            
            // Add brightness adjustment
            int new_value = pixel_value + brightness_delta;
            
            // Clamp to valid range using branching
            if (new_value > 255) {
                new_value = 255;
            }
            if (new_value < 0) {
                new_value = 0;
            }
            
            // Write back the adjusted value
            image[index] = (uint8_t)new_value;
        }
    }
}

// Utility function to get time in seconds
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// Benchmark function that measures throughput
void benchmark_brightness(uint8_t* image, int width, int height, int brightness_delta, int iterations) {
    double total_time = 0.0;
    uint8_t* temp_image = (uint8_t*)malloc(width * height);
    
    printf("Running benchmark with %d iterations...\n", iterations);
    
    for (int i = 0; i < iterations; i++) {
        // Make a copy to process
        memcpy(temp_image, image, width * height);
        
        // Time the brightness adjustment
        double start = get_time();
        adjust_brightness_naive(temp_image, width, height, brightness_delta);
        double end = get_time();
        
        double elapsed = end - start;
        total_time += elapsed;
        
        long long pixels_processed = (long long)width * height;
        double throughput = pixels_processed / elapsed;
        
        printf("  Iteration %d: %.6f seconds, %.2f Mpixels/sec\n", 
               i + 1, elapsed, throughput / 1e6);
    }
    
    // Calculate average statistics
    double avg_time = total_time / iterations;
    long long total_pixels = (long long)width * height;
    double avg_throughput = total_pixels / avg_time;
    
    printf("\nAverage Performance:\n");
    printf("  Time per iteration: %.6f seconds\n", avg_time);
    printf("  Throughput: %.2f Mpixels/sec\n", avg_throughput / 1e6);
    printf("  Throughput: %.0f pixels/sec\n", avg_throughput);
    
    free(temp_image);
}

// Read raw binary image file
uint8_t* read_image(const char* filename, int* width, int* height) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open input file %s\n", filename);
        return NULL;
    }
    
    // Read width and height from first 8 bytes
    if (fread(width, sizeof(int), 1, file) != 1 ||
        fread(height, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Error: Cannot read image dimensions\n");
        fclose(file);
        return NULL;
    }
    
    // Allocate memory for image data
    int size = (*width) * (*height);
    uint8_t* image = (uint8_t*)malloc(size);
    if (!image) {
        fprintf(stderr, "Error: Cannot allocate memory for image\n");
        fclose(file);
        return NULL;
    }
    
    // Read image data
    if (fread(image, 1, size, file) != size) {
        fprintf(stderr, "Error: Cannot read image data\n");
        free(image);
        fclose(file);
        return NULL;
    }
    
    fclose(file);
    return image;
}

// Write raw binary image file
int write_image(const char* filename, uint8_t* image, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open output file %s\n", filename);
        return 0;
    }
    
    // Write dimensions
    fwrite(&width, sizeof(int), 1, file);
    fwrite(&height, sizeof(int), 1, file);
    
    // Write image data
    int size = width * height;
    if (fwrite(image, 1, size, file) != size) {
        fprintf(stderr, "Error: Cannot write image data\n");
        fclose(file);
        return 0;
    }
    
    fclose(file);
    return 1;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printf("Usage: %s <input_image> <output_image> <brightness_value> [benchmark_iterations]\n", argv[0]);
        printf("  brightness_value: integer value to add to each pixel (-255 to 255)\n");
        printf("  benchmark_iterations: optional, number of times to run for benchmarking (default: 0)\n");
        return 1;
    }
    
    const char* input_file = argv[1];
    const char* output_file = argv[2];
    int brightness_delta = atoi(argv[3]);
    int benchmark_iterations = (argc > 4) ? atoi(argv[4]) : 0;
    
    printf("Brightness Adjustment Tool (Naive Implementation)\n");
    printf("Input: %s\n", input_file);
    printf("Output: %s\n", output_file);
    printf("Brightness adjustment: %+d\n\n", brightness_delta);
    
    // Read input image
    int width, height;
    uint8_t* image = read_image(input_file, &width, &height);
    if (!image) {
        return 1;
    }
    
    printf("Image dimensions: %dx%d (%d pixels)\n\n", width, height, width * height);
    
    // Run benchmark if requested
    if (benchmark_iterations > 0) {
        benchmark_brightness(image, width, height, brightness_delta, benchmark_iterations);
    }
    
    // Process the image once for output
    double start = get_time();
    adjust_brightness_naive(image, width, height, brightness_delta);
    double end = get_time();
    
    double elapsed = end - start;
    long long total_pixels = (long long)width * height;
    double throughput = total_pixels / elapsed;
    
    printf("\nFinal processing:\n");
    printf("  Time: %.6f seconds\n", elapsed);
    printf("  Throughput: %.2f Mpixels/sec\n", throughput / 1e6);
    
    // Write output image
    if (!write_image(output_file, image, width, height)) {
        free(image);
        return 1;
    }
    
    printf("\nOutput written to %s\n", output_file);
    
    free(image);
    return 0;
}