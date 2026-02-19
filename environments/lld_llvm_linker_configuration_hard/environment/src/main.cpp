#include <iostream>
#include "operations.h"
#include "helpers.h"

int main() {
    std::cout << "Calculator Application" << std::endl;
    std::cout << "======================" << std::endl;
    
    // Perform addition
    double sum = add(5.0, 3.0);
    print_result("Addition (5 + 3)", sum);
    
    // Perform multiplication
    double product = multiply(4.0, 7.0);
    print_result("Multiplication (4 * 7)", product);
    
    // Perform subtraction
    double difference = subtract(10.0, 4.0);
    print_result("Subtraction (10 - 4)", difference);
    
    // Print completion message
    print_message("All calculations completed successfully!");
    
    return 0;
}