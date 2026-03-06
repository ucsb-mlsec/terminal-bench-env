#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <basename>\n", argv[0]);
        return 1;
    }
    
    char filename[256];
    snprintf(filename, sizeof(filename), "%s.c", argv[1]);
    
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("fopen");
        return 1;
    }
    
    // Simulate some generation work
    usleep(100000);  // 100ms delay to increase chance of race conditions
    
    fprintf(fp, "#include <stdio.h>\n\n");
    fprintf(fp, "int func_%s(void) {\n", argv[1]);
    fprintf(fp, "    printf(\"Function from %s\\n\");\n", argv[1]);
    
    if (strcmp(argv[1], "generated1") == 0) {
        fprintf(fp, "    return 42;\n");
    } else if (strcmp(argv[1], "generated2") == 0) {
        fprintf(fp, "    return 58;\n");
    } else {
        fprintf(fp, "    return 100;\n");
    }
    
    fprintf(fp, "}\n");
    
    fclose(fp);
    printf("Generated %s\n", filename);
    return 0;
}