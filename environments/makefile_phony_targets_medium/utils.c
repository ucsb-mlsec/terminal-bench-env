#include <stdio.h>
#include <string.h>
#include <ctype.h>

void process_text(char* filename) {
    printf("Processing text from: %s\n", filename);
    
    int word_count = 0;
    
    printf("Word count: %d\n", word_count);
}

int main(int argc, char* argv[]) {
    if (argc > 1) {
        process_text(argv[1]);
    }
    
    return 0;
}