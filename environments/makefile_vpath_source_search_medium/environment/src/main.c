#include <stdio.h>
#include <stdlib.h>

// External function declarations
int add(int a, int b);
int multiply(int a, int b);
void process_string(char* str);
void print_message(void);

int main(void) {
    printf("Starting program...\n");
    
    print_message();
    
    int x = 10, y = 20;
    
    int sum = add(x, y);
    printf("Add result: %d\n", sum);
    
    int product = multiply(x, y);
    printf("Multiply result: %d\n", product);
    
    char text[] = "Hello World";
    process_string(text);
    
    printf("Program completed successfully\n");
    
    return 0;
}