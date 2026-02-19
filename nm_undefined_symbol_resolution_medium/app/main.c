#include <stdio.h>

/* External functions from utility library */
int calculate_sum(int a, int b);
void print_message(const char* msg);

int main() {
    int result;
    
    printf("Starting application...\n");
    
    /* Call utility function to print a message */
    print_message("Hello from the utility library!");
    
    /* Call utility function to calculate sum */
    result = calculate_sum(15, 27);
    printf("The sum of 15 and 27 is: %d\n", result);
    
    /* Another calculation */
    result = calculate_sum(100, 250);
    printf("The sum of 100 and 250 is: %d\n", result);
    
    print_message("Application completed successfully!");
    
    return 0;
}