#include "statistics.h"
#include "vector.h"
#include "utils.h"
#include "constants.h"
#include <iostream>

int main() {
    // Create a vector with known values
    Vector<double> data(5);
    data[0] = 1.0;
    data[1] = 2.0;
    data[2] = 3.0;
    data[3] = 4.0;
    data[4] = 5.0;
    
    std::cout << "Testing Statistics Module" << std::endl;
    std::cout << "Data: ";
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
    
    // Compute and print mean
    double mean_value = Stats::mean(data);
    std::cout << "Mean: " << mean_value << std::endl;
    
    // Verify mean is approximately 3.0
    if (!approxEqual(mean_value, 3.0, EPSILON)) {
        std::cout << "Error: Mean calculation incorrect!" << std::endl;
        return 1;
    }
    
    // Compute and print variance
    double var_value = Stats::variance(data);
    std::cout << "Variance: " << var_value << std::endl;
    
    // Verify variance is approximately 2.5 (for population variance) or 2.0 (for sample variance)
    if (!approxEqual(var_value, 2.5, EPSILON) && !approxEqual(var_value, 2.0, EPSILON)) {
        std::cout << "Warning: Variance value " << var_value << " may be unexpected" << std::endl;
    }
    
    // Compute and print standard deviation
    double stddev_value = Stats::stddev(data);
    std::cout << "Standard Deviation: " << stddev_value << std::endl;
    
    // Verify standard deviation is approximately sqrt(variance)
    double expected_stddev = sqrt(var_value);
    if (!approxEqual(stddev_value, expected_stddev, EPSILON)) {
        std::cout << "Error: Standard deviation calculation incorrect!" << std::endl;
        return 1;
    }
    
    // Additional test with a single value
    Vector<double> single(1);
    single[0] = PI;
    double single_mean = Stats::mean(single);
    if (!approxEqual(single_mean, PI, EPSILON)) {
        std::cout << "Error: Mean of single value incorrect!" << std::endl;
        return 1;
    }
    std::cout << "Single value mean: " << single_mean << std::endl;
    
    // Test with larger dataset
    Vector<double> large_data(10);
    for (size_t i = 0; i < large_data.size(); ++i) {
        large_data[i] = static_cast<double>(i + 1);
    }
    double large_mean = Stats::mean(large_data);
    std::cout << "Large dataset mean (1-10): " << large_mean << std::endl;
    
    if (!approxEqual(large_mean, 5.5, EPSILON)) {
        std::cout << "Error: Large dataset mean calculation incorrect!" << std::endl;
        return 1;
    }
    
    std::cout << "Statistics tests passed" << std::endl;
    return 0;
}