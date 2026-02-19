#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * File Compressor - Run-Length Encoding (RLE) Implementation
 * 
 * This program implements a basic file compression algorithm using
 * run-length encoding. RLE works by replacing sequences of identical
 * bytes with a count followed by the byte value.
 * 
 * Format: [count][byte] where count is 1-255
 * For sequences longer than 255, multiple count-byte pairs are used.
 */

#define MAX_RUN_LENGTH 255
#define BUFFER_SIZE 4096

/*
 * compress_file: Compresses input file using RLE algorithm
 * 
 * Reads the input file and writes compressed output.
 * Consecutive identical bytes are encoded as: count + byte value
 * 
 * Returns: 0 on success, -1 on failure
 */
int compress_file(const char *input_path, const char *output_path) {
    FILE *input = fopen(input_path, "rb");
    if (!input) {
        fprintf(stderr, "Error: Cannot open input file '%s'\n", input_path);
        return -1;
    }

    FILE *output = fopen(output_path, "wb");
    if (!output) {
        fprintf(stderr, "Error: Cannot create output file '%s'\n", output_path);
        fclose(input);
        return -1;
    }

    int current_byte, next_byte;
    int run_count;
    long original_size = 0;
    long compressed_size = 0;

    current_byte = fgetc(input);
    if (current_byte == EOF) {
        fprintf(stderr, "Warning: Input file is empty\n");
        fclose(input);
        fclose(output);
        return 0;
    }

    while (current_byte != EOF) {
        run_count = 1;
        original_size++;

        /* Count consecutive identical bytes */
        while ((next_byte = fgetc(input)) != EOF && 
               next_byte == current_byte && 
               run_count < MAX_RUN_LENGTH) {
            run_count++;
            original_size++;
        }

        /* Write count and byte value */
        fputc(run_count, output);
        fputc(current_byte, output);
        compressed_size += 2;

        current_byte = next_byte;
    }

    printf("Compression complete:\n");
    printf("  Original size: %ld bytes\n", original_size);
    printf("  Compressed size: %ld bytes\n", compressed_size);
    printf("  Ratio: %.2f%%\n", (compressed_size * 100.0) / original_size);

    fclose(input);
    fclose(output);
    return 0;
}

/*
 * decompress_file: Decompresses RLE-encoded file
 * 
 * Reads compressed file and restores original content.
 * Each pair of bytes represents: count + original byte value
 * 
 * Returns: 0 on success, -1 on failure
 */
int decompress_file(const char *input_path, const char *output_path) {
    FILE *input = fopen(input_path, "rb");
    if (!input) {
        fprintf(stderr, "Error: Cannot open input file '%s'\n", input_path);
        return -1;
    }

    FILE *output = fopen(output_path, "wb");
    if (!output) {
        fprintf(stderr, "Error: Cannot create output file '%s'\n", output_path);
        fclose(input);
        return -1;
    }

    int count, byte_value;
    long decompressed_size = 0;

    while ((count = fgetc(input)) != EOF) {
        byte_value = fgetc(input);
        
        if (byte_value == EOF) {
            fprintf(stderr, "Error: Unexpected end of compressed file\n");
            fclose(input);
            fclose(output);
            return -1;
        }

        /* Write the byte 'count' times */
        for (int i = 0; i < count; i++) {
            fputc(byte_value, output);
            decompressed_size++;
        }
    }

    printf("Decompression complete:\n");
    printf("  Decompressed size: %ld bytes\n", decompressed_size);

    fclose(input);
    fclose(output);
    return 0;
}

/*
 * print_usage: Display program usage information
 */
void print_usage(const char *program_name) {
    printf("File Compressor - RLE Implementation\n");
    printf("Usage: %s <mode> <input_file> <output_file>\n\n", program_name);
    printf("Modes:\n");
    printf("  -c, --compress     Compress input file\n");
    printf("  -d, --decompress   Decompress input file\n");
    printf("  -h, --help         Show this help message\n\n");
    printf("Examples:\n");
    printf("  %s -c data.txt data.rle\n", program_name);
    printf("  %s -d data.rle data_restored.txt\n", program_name);
}

/*
 * main: Entry point for the compression utility
 */
int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    /* Handle help option */
    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        print_usage(argv[0]);
        return 0;
    }

    /* Check for correct number of arguments */
    if (argc != 4) {
        fprintf(stderr, "Error: Invalid number of arguments\n\n");
        print_usage(argv[0]);
        return 1;
    }

    const char *mode = argv[1];
    const char *input_file = argv[2];
    const char *output_file = argv[3];

    /* Perform compression */
    if (strcmp(mode, "-c") == 0 || strcmp(mode, "--compress") == 0) {
        printf("Compressing '%s' to '%s'...\n", input_file, output_file);
        if (compress_file(input_file, output_file) != 0) {
            fprintf(stderr, "Compression failed\n");
            return 1;
        }
        printf("Success!\n");
        return 0;
    }
    
    /* Perform decompression */
    if (strcmp(mode, "-d") == 0 || strcmp(mode, "--decompress") == 0) {
        printf("Decompressing '%s' to '%s'...\n", input_file, output_file);
        if (decompress_file(input_file, output_file) != 0) {
            fprintf(stderr, "Decompression failed\n");
            return 1;
        }
        printf("Success!\n");
        return 0;
    }

    /* Unknown mode */
    fprintf(stderr, "Error: Unknown mode '%s'\n\n", mode);
    print_usage(argv[0]);
    return 1;
}