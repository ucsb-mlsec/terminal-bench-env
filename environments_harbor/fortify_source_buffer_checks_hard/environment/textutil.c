#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_BUFFER 256
#define OUTPUT_SIZE 512

void reverse_string(char *str) {
    int len = strlen(str);
    char temp[MAX_BUFFER];
    int i;
    
    strcpy(temp, str);
    
    for (i = 0; i < len; i++) {
        str[i] = temp[len - 1 - i];
    }
}

void to_uppercase(char *str) {
    int i;
    for (i = 0; str[i] != '\0'; i++) {
        if (str[i] >= 'a' && str[i] <= 'z') {
            str[i] = str[i] - 32;
        }
    }
}

void to_lowercase(char *str) {
    int i;
    for (i = 0; str[i] != '\0'; i++) {
        if (str[i] >= 'A' && str[i] <= 'Z') {
            str[i] = str[i] + 32;
        }
    }
}

void concatenate_strings(char *dest, char *src1, char *src2, char *src3) {
    strcpy(dest, src1);
    strcat(dest, " ");
    strcat(dest, src2);
    strcat(dest, " ");
    strcat(dest, src3);
}

void format_output(char *output, char *prefix, char *content, int count) {
    sprintf(output, "%s: %s (processed %d times)", prefix, content, count);
}

void duplicate_string(char *dest, char *src, int times) {
    char temp[MAX_BUFFER];
    int i;
    
    strcpy(dest, "");
    strcpy(temp, src);
    
    for (i = 0; i < times; i++) {
        strcat(dest, temp);
    }
}

void copy_with_prefix(char *dest, char *prefix, char *src) {
    char buffer[MAX_BUFFER];
    
    strcpy(buffer, prefix);
    strcat(buffer, src);
    memcpy(dest, buffer, strlen(buffer) + 1);
}

int main(int argc, char *argv[]) {
    char input_buffer[MAX_BUFFER];
    char output_buffer[OUTPUT_SIZE];
    char temp_buffer[MAX_BUFFER];
    char result[OUTPUT_SIZE];
    int operation = 0;
    int i;
    
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <operation> <text> [additional args]\n", argv[0]);
        fprintf(stderr, "Operations:\n");
        fprintf(stderr, "  1 - Reverse string\n");
        fprintf(stderr, "  2 - Convert to uppercase\n");
        fprintf(stderr, "  3 - Convert to lowercase\n");
        fprintf(stderr, "  4 - Concatenate arguments\n");
        fprintf(stderr, "  5 - Duplicate string\n");
        return 1;
    }
    
    operation = atoi(argv[1]);
    strcpy(input_buffer, argv[2]);
    
    switch(operation) {
        case 1:
            strcpy(temp_buffer, input_buffer);
            reverse_string(temp_buffer);
            sprintf(output_buffer, "Reversed: %s", temp_buffer);
            printf("%s\n", output_buffer);
            break;
            
        case 2:
            strcpy(temp_buffer, input_buffer);
            to_uppercase(temp_buffer);
            format_output(output_buffer, "Uppercase", temp_buffer, 1);
            printf("%s\n", output_buffer);
            break;
            
        case 3:
            strcpy(temp_buffer, input_buffer);
            to_lowercase(temp_buffer);
            copy_with_prefix(output_buffer, "Lowercase: ", temp_buffer);
            printf("%s\n", output_buffer);
            break;
            
        case 4:
            if (argc >= 5) {
                concatenate_strings(output_buffer, argv[2], argv[3], argv[4]);
                printf("Concatenated: %s\n", output_buffer);
            } else {
                fprintf(stderr, "Operation 4 requires at least 3 text arguments\n");
                return 1;
            }
            break;
            
        case 5:
            if (argc >= 4) {
                int times = atoi(argv[3]);
                duplicate_string(output_buffer, input_buffer, times);
                printf("Duplicated: %s\n", output_buffer);
            } else {
                fprintf(stderr, "Operation 5 requires a count argument\n");
                return 1;
            }
            break;
            
        default:
            fprintf(stderr, "Invalid operation: %d\n", operation);
            return 1;
    }
    
    strcpy(result, "Processing complete for input: ");
    strcat(result, input_buffer);
    
    memcpy(temp_buffer, result, strlen(result) + 1);
    
    printf("%s\n", temp_buffer);
    
    return 0;
}