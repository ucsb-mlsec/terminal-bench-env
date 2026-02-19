#include <stdio.h>
#include <stdlib.h>

/* Legacy calculator main program with various issues */

int main(int argc, char *argv[]) {
    int result;
    int unused_var;
    float price = 19.99;
    
    printf("Calculator Program Starting...\n");
    
    /* Using uninitialized variable */
    printf("Uninitialized result value: %d\n", result);
    
    /* Implicit function declaration - add() not declared */
    result = add(5, 3);
    printf("5 + 3 = %d\n", result);
    
    /* Format specifier mismatch - %d for float */
    printf("Price: %d\n", price);
    
    /* Implicit function declaration - multiply() not declared */
    result = multiply(4, 2);
    printf("4 * 2 = %d\n", result);
    
    /* Variable shadowing */
    {
        int result;  /* shadows outer result variable */
        result = 100;
        printf("Inner result: %d\n", result);
    }
    
    /* Implicit function declaration - helper_function() not declared */
    helper_function();
    
    /* Unused parameter argc - argv is used below but argc is not */
    if (argv[0] != NULL) {
        printf("Program name: %s\n", argv[0]);
    }
    
    printf("Calculator Program Finished.\n");
    
    return 0;
}