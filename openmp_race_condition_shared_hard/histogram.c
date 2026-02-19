#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <dirent.h>
#include <string.h>

#define NUM_BINS 10
#define BIN_WIDTH 10.0

int histogram[NUM_BINS] = {0};

void process_file(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        return;
    }
    
    float values[1000];
    int count = 0;
    
    while (fscanf(fp, "%f", &values[count]) == 1) {
        count++;
    }
    fclose(fp);
    
    #pragma omp parallel for
    for (int i = 0; i < count; i++) {
        float value = values[i];
        if (value >= 0.0 && value <= 100.0) {
            int bin_index = (int)(value / BIN_WIDTH);
            if (bin_index >= NUM_BINS) {
                bin_index = NUM_BINS - 1;
            }
            histogram[bin_index]++;
        }
    }
}

int main() {
    DIR *dir;
    struct dirent *entry;
    
    dir = opendir("data/");
    if (!dir) {
        fprintf(stderr, "Error opening data directory\n");
        return 1;
    }
    
    while ((entry = readdir(dir)) != NULL) {
        if (strstr(entry->d_name, ".txt") != NULL) {
            char filepath[512];
            snprintf(filepath, sizeof(filepath), "data/%s", entry->d_name);
            process_file(filepath);
        }
    }
    closedir(dir);
    
    for (int i = 0; i < NUM_BINS; i++) {
        printf("Bin %d (%.1f-%.1f): %d\n", 
               i, 
               i * BIN_WIDTH, 
               (i + 1) * BIN_WIDTH, 
               histogram[i]);
    }
    
    return 0;
}