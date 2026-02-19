#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include "utils.h"
#include "math_ops.h"
#include "string_ops.h"
#include "data_processor.h"

int main() {
    std::cout << "=== C++ Project with Multiple Components ===" << std::endl;
    
    // Initialize utility components
    Utils::Logger logger;
    logger.log("Application started");
    
    // Test math operations
    std::cout << "\n--- Math Operations ---" << std::endl;
    MathOps::Calculator calc;
    double result1 = calc.add(15.5, 24.3);
    double result2 = calc.multiply(7.0, 8.0);
    double result3 = calc.power(2.0, 10.0);
    
    std::cout << "Addition result: " << result1 << std::endl;
    std::cout << "Multiplication result: " << result2 << std::endl;
    std::cout << "Power result: " << result3 << std::endl;
    
    // Compute statistics
    std::vector<double> numbers = {12.5, 45.3, 23.1, 67.8, 34.2, 89.1};
    double avg = MathOps::Statistics::average(numbers);
    double stddev = MathOps::Statistics::standardDeviation(numbers);
    
    std::cout << "Average of dataset: " << avg << std::endl;
    std::cout << "Standard deviation: " << stddev << std::endl;
    
    // Test string operations
    std::cout << "\n--- String Operations ---" << std::endl;
    StringOps::TextProcessor textProc;
    
    std::string input = "Hello World From C++ Project";
    std::string upper = textProc.toUpperCase(input);
    std::string lower = textProc.toLowerCase(input);
    std::string reversed = textProc.reverse(input);
    
    std::cout << "Original: " << input << std::endl;
    std::cout << "Uppercase: " << upper << std::endl;
    std::cout << "Lowercase: " << lower << std::endl;
    std::cout << "Reversed: " << reversed << std::endl;
    
    int wordCount = StringOps::countWords(input);
    std::cout << "Word count: " << wordCount << std::endl;
    
    // Test data processing
    std::cout << "\n--- Data Processing ---" << std::endl;
    DataProcessor::DataSet dataset;
    
    dataset.addEntry("sensor1", 23.5);
    dataset.addEntry("sensor2", 45.7);
    dataset.addEntry("sensor3", 67.2);
    dataset.addEntry("sensor4", 12.8);
    dataset.addEntry("sensor5", 89.3);
    
    std::cout << "Dataset size: " << dataset.size() << std::endl;
    std::cout << "Dataset average: " << dataset.getAverage() << std::endl;
    std::cout << "Dataset maximum: " << dataset.getMaximum() << std::endl;
    std::cout << "Dataset minimum: " << dataset.getMinimum() << std::endl;
    
    // Process and filter data
    DataProcessor::FilterEngine filter;
    auto filteredData = filter.filterByThreshold(numbers, 30.0);
    std::cout << "Filtered data points (>30): " << filteredData.size() << std::endl;
    
    // Final summary
    std::cout << "\n--- Summary ---" << std::endl;
    logger.log("All components tested successfully");
    logger.log("Application completed");
    
    Utils::Timer timer;
    std::cout << "Total operations completed" << std::endl;
    std::cout << "Execution time: " << timer.elapsed() << " ms" << std::endl;
    
    return 0;
}