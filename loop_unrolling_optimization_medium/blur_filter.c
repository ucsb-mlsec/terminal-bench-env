#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define MAX_PIXEL_VALUE 255

typedef struct {
    int width;
    int height;
    int max_value;
    int **data;
} PGMImage;

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

PGMImage* allocate_image(int width, int height) {
    PGMImage *img = (PGMImage*)malloc(sizeof(PGMImage));
    img->width = width;
    img->height = height;
    img->max_value = MAX_PIXEL_VALUE;
    
    img->data = (int**)malloc(height * sizeof(int*));
    for (int i = 0; i < height; i++) {
        img->data[i] = (int*)malloc(width * sizeof(int));
    }
    
    return img;
}

void free_image(PGMImage *img) {
    if (img) {
        for (int i = 0; i < img->height; i++) {
            free(img->data[i]);
        }
        free(img->data);
        free(img);
    }
}

PGMImage* read_pgm(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return NULL;
    }
    
    char magic[3];
    fscanf(fp, "%s", magic);
    
    if (magic[0] != 'P' || magic[1] != '2') {
        fprintf(stderr, "Error: Invalid PGM format (expected P2)\n");
        fclose(fp);
        return NULL;
    }
    
    int width, height, max_value;
    fscanf(fp, "%d %d", &width, &height);
    fscanf(fp, "%d", &max_value);
    
    PGMImage *img = allocate_image(width, height);
    img->max_value = max_value;
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            fscanf(fp, "%d", &img->data[i][j]);
        }
    }
    
    fclose(fp);
    return img;
}

void write_pgm(const char *filename, PGMImage *img) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot write to file %s\n", filename);
        return;
    }
    
    fprintf(fp, "P2\n");
    fprintf(fp, "%d %d\n", img->width, img->height);
    fprintf(fp, "%d\n", img->max_value);
    
    for (int i = 0; i < img->height; i++) {
        for (int j = 0; j < img->width; j++) {
            fprintf(fp, "%d ", img->data[i][j]);
        }
        fprintf(fp, "\n");
    }
    
    fclose(fp);
}

PGMImage* apply_blur_filter(PGMImage *input) {
    PGMImage *output = allocate_image(input->width, input->height);
    output->max_value = input->max_value;
    
    // Process interior pixels with 3x3 blur kernel
    for (int i = 0; i < input->height; i++) {
        for (int j = 0; j < input->width; j++) {
            int sum = 0;
            int count = 0;
            
            // Apply 3x3 averaging filter
            for (int di = -1; di <= 1; di++) {
                for (int dj = -1; dj <= 1; dj++) {
                    int ni = i + di;
                    int nj = j + dj;
                    
                    // Check boundaries
                    if (ni >= 0 && ni < input->height && nj >= 0 && nj < input->width) {
                        sum += input->data[ni][nj];
                        count++;
                    }
                }
            }
            
            output->data[i][j] = sum / count;
        }
    }
    
    return output;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input.pgm>\n", argv[0]);
        return 1;
    }
    
    const char *input_filename = argv[1];
    const char *output_filename = "output.pgm";
    
    // Read input image
    PGMImage *input = read_pgm(input_filename);
    if (!input) {
        return 1;
    }
    
    // Start timing
    double start_time = get_time();
    
    // Apply blur filter (run multiple times to make timing more measurable)
    PGMImage *output = NULL;
    for (int iteration = 0; iteration < 50; iteration++) {
        if (output) {
            free_image(output);
        }
        output = apply_blur_filter(input);
    }
    
    // End timing
    double end_time = get_time();
    double elapsed_time = end_time - start_time;
    
    // Write output image
    write_pgm(output_filename, output);
    
    // Print processing time
    printf("Processing time: %.2f seconds\n", elapsed_time);
    
    // Clean up
    free_image(input);
    free_image(output);
    
    return 0;
}