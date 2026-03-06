#include <iostream>
#include <string>
#include <cstring>
#include <cmath>

// Utility functions for scientific computing application
// No OpenMP directives - these are sequential helper functions

void print_array(const double* array, int size, const std::string& label) {
    std::cout << label << ": ";
    for (int i = 0; i < size && i < 10; i++) {
        std::cout << array[i] << " ";
    }
    if (size > 10) {
        std::cout << "... (" << size << " total elements)";
    }
    std::cout << std::endl;
}

double* allocate_memory(int size) {
    if (size <= 0) {
        std::cerr << "Error: Invalid allocation size " << size << std::endl;
        return nullptr;
    }
    
    double* ptr = new double[size];
    if (ptr == nullptr) {
        std::cerr << "Error: Memory allocation failed for size " << size << std::endl;
        return nullptr;
    }
    
    std::memset(ptr, 0, size * sizeof(double));
    return ptr;
}

void free_memory(double* ptr) {
    if (ptr != nullptr) {
        delete[] ptr;
    }
}

void initialize_data(double* array, int size, double init_value) {
    if (array == nullptr) {
        std::cerr << "Error: Cannot initialize null array" << std::endl;
        return;
    }
    
    for (int i = 0; i < size; i++) {
        array[i] = init_value + static_cast<double>(i) * 0.1;
    }
}

bool validate_results(const double* computed, const double* expected, int size, double tolerance) {
    if (computed == nullptr || expected == nullptr) {
        std::cerr << "Error: Cannot validate null arrays" << std::endl;
        return false;
    }
    
    bool valid = true;
    int error_count = 0;
    double max_error = 0.0;
    
    for (int i = 0; i < size; i++) {
        double diff = std::fabs(computed[i] - expected[i]);
        if (diff > tolerance) {
            error_count++;
            valid = false;
            if (diff > max_error) {
                max_error = diff;
            }
        }
    }
    
    if (!valid) {
        std::cerr << "Validation failed: " << error_count << " errors found" << std::endl;
        std::cerr << "Maximum error: " << max_error << std::endl;
    } else {
        std::cout << "Validation passed: all results within tolerance " << tolerance << std::endl;
    }
    
    return valid;
}

double compute_checksum(const double* array, int size) {
    if (array == nullptr || size <= 0) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += array[i];
    }
    return sum;
}

void copy_array(const double* source, double* destination, int size) {
    if (source == nullptr || destination == nullptr) {
        std::cerr << "Error: Cannot copy null arrays" << std::endl;
        return;
    }
    
    std::memcpy(destination, source, size * sizeof(double));
}