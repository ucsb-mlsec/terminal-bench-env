#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <image_file>\n", argv[0]);
        return 1;
    }

    FILE *file = fopen(argv[1], "r");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", argv[1]);
        return 1;
    }

    int width, height;
    if (fscanf(file, "%d %d", &width, &height) != 2) {
        fprintf(stderr, "Error: Invalid file format\n");
        fclose(file);
        return 1;
    }

    // Variable Length Array allocated on stack
    unsigned char image[height][width];

    // Read pixel values from file
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int pixel;
            if (fscanf(file, "%d", &pixel) != 1) {
                fprintf(stderr, "Error: Failed to read pixel data\n");
                fclose(file);
                return 1;
            }
            image[i][j] = (unsigned char)pixel;
        }
    }

    // Process the image - calculate sum for demonstration
    unsigned long long sum = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            sum += image[i][j];
        }
    }

    double average = (double)sum / (width * height);

    printf("Successfully processed image: %dx%d pixels\n", width, height);
    printf("Average pixel value: %.2f\n", average);

    fclose(file);
    return 0;
}