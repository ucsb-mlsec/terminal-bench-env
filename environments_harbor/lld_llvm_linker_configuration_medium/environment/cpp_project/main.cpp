#include <iostream>
#include "math_ops.h"
#include "string_utils.h"

int main() {
    // Test math operations
    int result = add(5, 3);
    std::cout << "5 + 3 = " << result << std::endl;
    
    // Test string utilities
    std::string test_str = "hello";
    std::string upper = toUpper(test_str);
    std::cout << "Uppercase: " << upper << std::endl;
    
    int len = getLength(test_str);
    std::cout << "Length: " << len << std::endl;
    
    std::cout << "All tests completed successfully!" << std::endl;
    return 0;
}