#include <stdio.h>
#include <string.h>

void parse_input(char* filename) {
    printf("Parsing file: %s\n", filename);
    
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file\n");
        return;
    }
    
    fclose(file);
}