#include <stdio.h>
#include <string.h>

// Function prototypes for math library
int add(int a, int b);
int multiply(int a, int b);
int math_process(int value);

// Function prototypes for string library
void string_concat(char* dest, const char* src1, const char* src2);
int string_len(const char* str);
void string_process(const char* str);

int main() {
    printf("Testing Math Library Functions:\n");
    printf("================================\n");
    
    // Call add from math library
    int sum = add(5, 3);
    printf("add(5, 3) = %d\n", sum);
    
    // Call multiply from math library
    int product = multiply(4, 7);
    printf("multiply(4, 7) = %d\n", product);
    
    // Call process with integer argument (math version)
    int processed_int = math_process(10);
    printf("math_process(10) = %d\n", processed_int);
    
    printf("\nTesting String Library Functions:\n");
    printf("==================================\n");
    
    // Call string_concat from string library
    char buffer[100];
    string_concat(buffer, "Hello", " World");
    printf("string_concat result: %s\n", buffer);
    
    // Call string_len from string library
    const char* test_string = "Testing";
    int length = string_len(test_string);
    printf("string_len(\"%s\") = %d\n", test_string, length);
    
    // Call process with string argument (string version)
    string_process("Sample String");
    
    printf("\nAll tests completed successfully!\n");
    
    return 0;
}