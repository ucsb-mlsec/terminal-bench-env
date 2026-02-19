#include <stdio.h>
#include <stdlib.h>

int main() {
    FILE *fp;
    
    fp = fopen("generated.h", "w");
    if (fp == NULL) {
        fprintf(stderr, "Error: Could not create generated.h\n");
        return 1;
    }
    
    fprintf(fp, "/* Auto-generated header file - do not edit manually */\n\n");
    fprintf(fp, "#ifndef GENERATED_H\n");
    fprintf(fp, "#define GENERATED_H\n\n");
    
    fprintf(fp, "#define VERSION_NUMBER 100\n");
    fprintf(fp, "#define BUFFER_SIZE 1024\n");
    fprintf(fp, "#define MAX_ITEMS 50\n");
    fprintf(fp, "#define CONFIG_VALUE 42\n\n");
    
    fprintf(fp, "#endif /* GENERATED_H */\n");
    
    if (fclose(fp) != 0) {
        fprintf(stderr, "Error: Could not close generated.h\n");
        return 1;
    }
    
    printf("Generated header file: generated.h\n");
    
    return 0;
}