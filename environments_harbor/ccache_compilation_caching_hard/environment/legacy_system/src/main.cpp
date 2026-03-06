#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include "compute_engine.h"
#include "data_processor.h"
#include "io_handler.h"
#include "validator.h"

/**
 * Main entry point for the legacy system application.
 * This application demonstrates the integration of various subsystems:
 * - Compute Engine: Handles mathematical computations
 * - Data Processor: Processes and transforms data structures
 * - IO Handler: Manages input/output operations
 * - Validator: Ensures data integrity and correctness
 */
int main(int argc, char* argv[]) {
    try {
        // Initialize the IO Handler for managing file operations
        std::unique_ptr<IOHandler> ioHandler = std::make_unique<IOHandler>();
        ioHandler->initialize();
        
        // Create sample data for processing
        std::vector<double> inputData = {1.0, 2.5, 3.7, 4.2, 5.9};
        ioHandler->setData(inputData);
        
        // Initialize the Compute Engine for heavy computational tasks
        ComputeEngine computeEngine;
        computeEngine.configure();
        
        // Process the data through the compute engine
        std::vector<double> computedResults = computeEngine.process(inputData);
        
        // Initialize the Data Processor for transformation operations
        DataProcessor dataProcessor;
        dataProcessor.setup();
        
        // Transform the computed results
        std::vector<double> processedData = dataProcessor.transform(computedResults);
        
        // Write intermediate results
        ioHandler->writeResults(processedData);
        
        // Initialize the Validator to ensure data integrity
        Validator validator;
        validator.initialize();
        
        // Validate the processed data meets all requirements
        bool isValid = validator.validate(processedData);
        
        if (!isValid) {
            std::cerr << "Validation failed for processed data" << std::endl;
            return 1;
        }
        
        // Perform additional validation on computation results
        bool computationValid = validator.checkComputations(computedResults);
        
        if (!computationValid) {
            std::cerr << "Computation validation failed" << std::endl;
            return 1;
        }
        
        // All operations completed successfully
        std::cout << "VALIDATION_SUCCESS" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}