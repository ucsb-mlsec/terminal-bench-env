#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <cstring>
#include <pthread.h>
#include <unistd.h>

#include "parser.h"
#include "filter.h"
#include "aggregator.h"
#include "transformer.h"
#include "validator.h"
#include "writer.h"
#include "config.h"

struct ThreadData {
    std::vector<std::vector<std::string>> data;
    int thread_id;
    int num_threads;
    std::vector<std::vector<std::string>> result;
};

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " [OPTIONS]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -i <file>    Input CSV file (required)" << std::endl;
    std::cerr << "  -o <file>    Output CSV file (required)" << std::endl;
    std::cerr << "  -c <file>    Configuration file (optional)" << std::endl;
    std::cerr << "  -t <num>     Number of threads (default: 4)" << std::endl;
    std::cerr << "  -f <filter>  Filter expression (optional)" << std::endl;
    std::cerr << "  -v           Validate data" << std::endl;
    std::cerr << "  -h           Show this help message" << std::endl;
}

void* process_chunk(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);
    
    int chunk_size = data->data.size() / data->num_threads;
    int start_idx = data->thread_id * chunk_size;
    int end_idx = (data->thread_id == data->num_threads - 1) ? 
                   data->data.size() : (start_idx + chunk_size);
    
    for (int i = start_idx; i < end_idx; i++) {
        std::vector<std::string> row = data->data[i];
        
        // Transform the row
        std::vector<std::string> transformed = transform_row(row);
        
        // Apply filters if needed
        if (filter_row(transformed)) {
            data->result.push_back(transformed);
        }
    }
    
    return nullptr;
}

bool read_csv_file(const std::string& filename, 
                   std::vector<std::vector<std::string>>& data) {
    std::ifstream file(filename.c_str());
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open input file: " << filename << std::endl;
        return false;
    }
    
    std::string line;
    int line_number = 0;
    
    while (std::getline(file, line)) {
        line_number++;
        
        if (line.empty()) {
            continue;
        }
        
        std::vector<std::string> row = parse_csv_line(line);
        
        if (row.empty()) {
            std::cerr << "Warning: Empty row at line " << line_number << std::endl;
            continue;
        }
        
        data.push_back(row);
    }
    
    file.close();
    
    std::cout << "Read " << data.size() << " rows from " << filename << std::endl;
    return true;
}

bool write_csv_file(const std::string& filename,
                    const std::vector<std::vector<std::string>>& data) {
    if (!write_data_to_file(filename, data)) {
        std::cerr << "Error: Cannot write to output file: " << filename << std::endl;
        return false;
    }
    
    std::cout << "Wrote " << data.size() << " rows to " << filename << std::endl;
    return true;
}

int process_data_parallel(std::vector<std::vector<std::string>>& input_data,
                         std::vector<std::vector<std::string>>& output_data,
                         int num_threads) {
    if (input_data.empty()) {
        std::cerr << "Error: No data to process" << std::endl;
        return -1;
    }
    
    if (num_threads <= 0) {
        num_threads = 1;
    }
    
    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadData> thread_data(num_threads);
    
    // Initialize thread data
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].data = input_data;
        thread_data[i].thread_id = i;
        thread_data[i].num_threads = num_threads;
    }
    
    // Create threads
    for (int i = 0; i < num_threads; i++) {
        int rc = pthread_create(&threads[i], nullptr, process_chunk, 
                               &thread_data[i]);
        if (rc != 0) {
            std::cerr << "Error: pthread_create failed with code " << rc << std::endl;
            return -1;
        }
    }
    
    // Join threads
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
    }
    
    // Merge results
    for (int i = 0; i < num_threads; i++) {
        output_data.insert(output_data.end(),
                          thread_data[i].result.begin(),
                          thread_data[i].result.end());
    }
    
    return 0;
}

void calculate_statistics(const std::vector<std::vector<std::string>>& data) {
    if (data.empty()) {
        return;
    }
    
    std::cout << "\nData Statistics:" << std::endl;
    std::cout << "  Total rows: " << data.size() << std::endl;
    
    if (!data.empty() && !data[0].empty()) {
        std::cout << "  Columns: " << data[0].size() << std::endl;
    }
    
    // Calculate some aggregate statistics
    AggregateResult result = calculate_aggregates(data);
    std::cout << "  Sum: " << result.sum << std::endl;
    std::cout << "  Average: " << result.average << std::endl;
    std::cout << "  Min: " << result.min << std::endl;
    std::cout << "  Max: " << result.max << std::endl;
    
    double std_dev = std::sqrt(result.variance);
    std::cout << "  Std Dev: " << std_dev << std::endl;
}

int main(int argc, char* argv[]) {
    std::string input_file;
    std::string output_file;
    std::string config_file;
    std::string filter_expr;
    int num_threads = 4;
    bool validate = false;
    
    // Parse command-line arguments
    int opt;
    while ((opt = getopt(argc, argv, "i:o:c:t:f:vh")) != -1) {
        switch (opt) {
            case 'i':
                input_file = optarg;
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'c':
                config_file = optarg;
                break;
            case 't':
                num_threads = std::atoi(optarg);
                break;
            case 'f':
                filter_expr = optarg;
                break;
            case 'v':
                validate = true;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    
    // Validate required arguments
    if (input_file.empty() || output_file.empty()) {
        std::cerr << "Error: Input and output files are required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    // Load configuration if provided
    if (!config_file.empty()) {
        if (!load_config(config_file)) {
            std::cerr << "Warning: Could not load config file: " 
                     << config_file << std::endl;
        } else {
            std::cout << "Loaded configuration from " << config_file << std::endl;
        }
    }
    
    // Set filter expression if provided
    if (!filter_expr.empty()) {
        set_filter_expression(filter_expr);
    }
    
    std::cout << "Starting CSV data processing..." << std::endl;
    std::cout << "Input: " << input_file << std::endl;
    std::cout << "Output: " << output_file << std::endl;
    std::cout << "Threads: " << num_threads << std::endl;
    
    // Read input data
    std::vector<std::vector<std::string>> input_data;
    if (!read_csv_file(input_file, input_data)) {
        return 1;
    }
    
    // Validate data if requested
    if (validate) {
        std::cout << "Validating input data..." << std::endl;
        ValidationResult validation = validate_data(input_data);
        
        if (!validation.is_valid) {
            std::cerr << "Error: Data validation failed" << std::endl;
            std::cerr << "  Errors: " << validation.error_count << std::endl;
            for (size_t i = 0; i < validation.errors.size(); i++) {
                std::cerr << "  - " << validation.errors[i] << std::endl;
            }
            return 1;
        }
        
        std::cout << "Data validation passed" << std::endl;
    }
    
    // Process data
    std::vector<std::vector<std::string>> output_data;
    int result = process_data_parallel(input_data, output_data, num_threads);
    
    if (result != 0) {
        std::cerr << "Error: Data processing failed" << std::endl;
        return 1;
    }
    
    std::cout << "Processing complete" << std::endl;
    
    // Calculate and display statistics
    calculate_statistics(output_data);
    
    // Write output data
    if (!write_csv_file(output_file, output_data)) {
        return 1;
    }
    
    std::cout << "Data processing completed successfully" << std::endl;
    
    return 0;
}