/*
 * Legacy Image Processing Utility
 * 
 * This utility reads image files and performs basic validation and processing.
 * Originally designed to work with command-line file arguments.
 * 
 * Usage: ./processor <image_file>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FILE_SIZE 10485760  // 10MB max file size
#define HEADER_SIZE 32

// Image format identifiers
typedef enum {
    FORMAT_UNKNOWN = 0,
    FORMAT_JPEG,
    FORMAT_PNG,
    FORMAT_GIF,
    FORMAT_BMP
} ImageFormat;

/*
 * Detect image format by examining magic bytes
 * Returns the detected format type
 */
ImageFormat detect_format(unsigned char *buffer, size_t size) {
    if (size < 4) {
        return FORMAT_UNKNOWN;
    }
    
    // Check JPEG magic bytes (FF D8 FF)
    if (buffer[0] == 0xFF && buffer[1] == 0xD8 && buffer[2] == 0xFF) {
        printf("[INFO] Detected JPEG format\n");
        return FORMAT_JPEG;
    }
    
    // Check PNG magic bytes (89 50 4E 47)
    if (buffer[0] == 0x89 && buffer[1] == 0x50 && 
        buffer[2] == 0x4E && buffer[3] == 0x47) {
        printf("[INFO] Detected PNG format\n");
        return FORMAT_PNG;
    }
    
    // Check GIF magic bytes (47 49 46)
    if (buffer[0] == 0x47 && buffer[1] == 0x49 && buffer[2] == 0x46) {
        printf("[INFO] Detected GIF format\n");
        return FORMAT_GIF;
    }
    
    // Check BMP magic bytes (42 4D)
    if (buffer[0] == 0x42 && buffer[1] == 0x4D) {
        printf("[INFO] Detected BMP format\n");
        return FORMAT_BMP;
    }
    
    printf("[WARNING] Unknown image format\n");
    return FORMAT_UNKNOWN;
}

/*
 * Validate image header for basic consistency
 * Returns 0 on success, -1 on failure
 */
int validate_header(unsigned char *buffer, size_t size, ImageFormat format) {
    if (format == FORMAT_UNKNOWN) {
        fprintf(stderr, "[ERROR] Cannot validate unknown format\n");
        return -1;
    }
    
    // Basic size validation
    if (size < HEADER_SIZE) {
        fprintf(stderr, "[ERROR] File too small to contain valid header\n");
        return -1;
    }
    
    // Format-specific validation
    switch (format) {
        case FORMAT_JPEG:
            // Check for valid JPEG marker
            if (size > 10 && buffer[3] >= 0xE0 && buffer[3] <= 0xEF) {
                printf("[INFO] JPEG header validation passed\n");
                return 0;
            }
            break;
            
        case FORMAT_PNG:
            // Check PNG signature continuation
            if (size > 8 && buffer[4] == 0x0D && buffer[5] == 0x0A && 
                buffer[6] == 0x1A && buffer[7] == 0x0A) {
                printf("[INFO] PNG header validation passed\n");
                return 0;
            }
            break;
            
        case FORMAT_GIF:
            // Check GIF version (87a or 89a)
            if (size > 6 && (buffer[3] == '8') && 
                (buffer[4] == '7' || buffer[4] == '9') && buffer[5] == 'a') {
                printf("[INFO] GIF header validation passed\n");
                return 0;
            }
            break;
            
        case FORMAT_BMP:
            // Basic BMP header size check
            if (size > 14) {
                printf("[INFO] BMP header validation passed\n");
                return 0;
            }
            break;
            
        default:
            break;
    }
    
    fprintf(stderr, "[ERROR] Header validation failed for detected format\n");
    return -1;
}

/*
 * Process image data - performs transformations and analysis
 * Returns 0 on success, -1 on failure
 */
int process_image(unsigned char *buffer, size_t size, ImageFormat format) {
    printf("[INFO] Processing image data (%zu bytes)...\n", size);
    
    // Simple byte analysis
    unsigned long sum = 0;
    for (size_t i = 0; i < size && i < 1024; i++) {
        sum += buffer[i];
    }
    
    printf("[INFO] Checksum of first 1KB: %lu\n", sum);
    
    // Simulate some processing based on format
    switch (format) {
        case FORMAT_JPEG:
            printf("[INFO] Applying JPEG-specific processing\n");
            // In a real utility, this would do actual JPEG operations
            break;
            
        case FORMAT_PNG:
            printf("[INFO] Applying PNG-specific processing\n");
            // In a real utility, this would do actual PNG operations
            break;
            
        case FORMAT_GIF:
            printf("[INFO] Applying GIF-specific processing\n");
            break;
            
        case FORMAT_BMP:
            printf("[INFO] Applying BMP-specific processing\n");
            break;
            
        default:
            printf("[INFO] Applying generic processing\n");
            break;
    }
    
    printf("[SUCCESS] Image processing completed\n");
    return 0;
}

/*
 * Read entire file into memory buffer
 * Returns buffer pointer on success, NULL on failure
 * Caller is responsible for freeing the buffer
 */
unsigned char* read_file(const char *filename, size_t *file_size) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "[ERROR] Cannot open file: %s\n", filename);
        return NULL;
    }
    
    // Get file size
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    if (size <= 0 || size > MAX_FILE_SIZE) {
        fprintf(stderr, "[ERROR] Invalid file size: %ld bytes\n", size);
        fclose(fp);
        return NULL;
    }
    
    printf("[INFO] Reading file: %s (%ld bytes)\n", filename, size);
    
    // Allocate buffer
    unsigned char *buffer = (unsigned char *)malloc(size);
    if (!buffer) {
        fprintf(stderr, "[ERROR] Memory allocation failed\n");
        fclose(fp);
        return NULL;
    }
    
    // Read file data
    size_t bytes_read = fread(buffer, 1, size, fp);
    fclose(fp);
    
    if (bytes_read != (size_t)size) {
        fprintf(stderr, "[ERROR] Failed to read complete file\n");
        free(buffer);
        return NULL;
    }
    
    *file_size = bytes_read;
    return buffer;
}

/*
 * Main entry point
 * Expects filename as command-line argument
 */
int main(int argc, char *argv[]) {
    printf("=== Image Processing Utility v1.0 ===\n\n");
    
    // Check command-line arguments
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <image_file>\n", argv[0]);
        return 1;
    }
    
    const char *filename = argv[1];
    size_t file_size = 0;
    
    // Read the image file
    unsigned char *buffer = read_file(filename, &file_size);
    if (!buffer) {
        return 2;
    }
    
    // Detect image format
    ImageFormat format = detect_format(buffer, file_size);
    
    // Validate header
    if (validate_header(buffer, file_size, format) != 0) {
        free(buffer);
        return 3;
    }
    
    // Process the image
    int result = process_image(buffer, file_size, format);
    
    // Cleanup
    free(buffer);
    
    if (result == 0) {
        printf("\n[SUCCESS] Image processing completed successfully\n");
        return 0;
    } else {
        fprintf(stderr, "\n[ERROR] Image processing failed\n");
        return 4;
    }
}