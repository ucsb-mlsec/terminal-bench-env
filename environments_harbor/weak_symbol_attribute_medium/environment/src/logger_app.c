#include <stdio.h>
#include <string.h>

extern void format_json(const char* message);
extern void format_xml(const char* message);

int main() {
    printf("Logger Application Starting...\n");
    
    printf("\nAttempting JSON formatting:\n");
    format_json("Test log message");
    
    printf("\nAttempting XML formatting:\n");
    format_xml("Test log message");
    
    printf("\nLogger Application Complete\n");
    
    return 0;
}