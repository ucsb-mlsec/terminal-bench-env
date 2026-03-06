#include "common.h"

int main() {
    // Print welcome message
    std::cout << "Calculator Application Started" << std::endl;
    std::cout << "===============================" << std::endl;
    
    // Create a vector with some numbers
    std::vector<int> numbers = {10, 20, 30, 40, 50};
    
    // Calculate sum using accumulation
    int sum = 0;
    for (const auto& num : numbers) {
        sum += num;
    }
    
    // Display results using string operations
    std::string result_message = "Sum of numbers: ";
    std::cout << result_message << sum << std::endl;
    
    // Calculate average
    double average = static_cast<double>(sum) / numbers.size();
    std::cout << "Average: " << average << std::endl;
    
    // Display individual numbers
    std::cout << "\nNumbers processed: ";
    for (size_t i = 0; i < numbers.size(); ++i) {
        std::cout << numbers[i];
        if (i < numbers.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;
    
    std::cout << "\nCalculation complete!" << std::endl;
    
    return 0;
}