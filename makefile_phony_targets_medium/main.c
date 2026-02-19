#include <stdio.h>
#include <stdlib.h>

extern void parse_input(char*);
extern void process_text(char*);

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: textproc <filename>\n");
        return 1;
    }
    
    parse_input(argv[1]);
    process_text(argv[1]);
    
    printf("Processing complete\n");
    
    return 0;
}