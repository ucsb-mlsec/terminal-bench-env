package com.legacy.services;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.logging.Logger;
import java.util.logging.Level;

/**
 * DataProcessor - Legacy service for processing binary data and parsing JSON
 * 
 * This service relies on native libraries for high-performance data processing
 * and JSON parsing operations. The following native libraries are required:
 * 
 * - libdata_proc_x64.so: Core data processing algorithms (x86_64 architecture)
 * - libjson_parser.so: Native JSON parsing library for performance-critical operations
 * - libutils_x86_64.so: Utility functions shared across native components
 * 
 * These libraries must be available in the java.library.path at runtime.
 * The JVM will automatically append platform-specific prefixes (lib) and 
 * suffixes (.so on Linux) when loading libraries via System.loadLibrary().
 * 
 * @author Legacy Systems Team
 * @version 2.1
 */
public class DataProcessor {
    
    private static final Logger LOGGER = Logger.getLogger(DataProcessor.class.getName());
    private static boolean librariesLoaded = false;
    
    static {
        try {
            // Load native libraries in dependency order
            // utils must be loaded first as other libraries depend on it
            System.loadLibrary("utils_x86_64");
            LOGGER.info("Loaded native library: utils_x86_64");
            
            System.loadLibrary("json_parser");
            LOGGER.info("Loaded native library: json_parser");
            
            System.loadLibrary("data_proc_x64");
            LOGGER.info("Loaded native library: data_proc_x64");
            
            librariesLoaded = true;
            LOGGER.info("All native libraries loaded successfully");
            
        } catch (UnsatisfiedLinkError e) {
            LOGGER.log(Level.SEVERE, "Failed to load native libraries", e);
            LOGGER.severe("java.library.path: " + System.getProperty("java.library.path"));
            librariesLoaded = false;
        }
    }
    
    /**
     * Processes binary data using native implementation
     * 
     * @param input Binary data to process
     * @return Processed data as String
     * @throws RuntimeException if native library is not loaded
     */
    public native String processData(byte[] input);
    
    /**
     * Parses JSON string using native high-performance parser
     * 
     * @param json JSON string to parse
     * @return Parsed object representation
     * @throws RuntimeException if parsing fails or library not loaded
     */
    public native Object parseJson(String json);
    
    /**
     * Validates data integrity using native CRC implementation
     * 
     * @param data Data to validate
     * @return true if data is valid, false otherwise
     */
    public native boolean validateData(byte[] data);
    
    /**
     * Compresses data using native compression algorithm
     * 
     * @param input Data to compress
     * @return Compressed byte array
     */
    public native byte[] compressData(byte[] input);
    
    /**
     * Checks if all required native libraries are loaded
     * 
     * @return true if libraries are loaded, false otherwise
     */
    public static boolean isNativeReady() {
        return librariesLoaded;
    }
    
    /**
     * Main entry point for DataProcessor service
     * 
     * @param args Command line arguments
     */
    public static void main(String[] args) {
        LOGGER.info("Starting DataProcessor service...");
        
        if (!isNativeReady()) {
            LOGGER.severe("Native libraries not loaded. Cannot start service.");
            LOGGER.severe("Please ensure java.library.path is configured correctly.");
            System.exit(1);
        }
        
        DataProcessor processor = new DataProcessor();
        
        try {
            // Test native library functionality
            byte[] testData = "Test data for processing".getBytes();
            LOGGER.info("Testing native data processing...");
            String result = processor.processData(testData);
            LOGGER.info("Processing result: " + result);
            
            // Test JSON parsing
            String jsonTest = "{\"status\":\"ok\",\"message\":\"test\"}";
            LOGGER.info("Testing native JSON parsing...");
            Object jsonResult = processor.parseJson(jsonTest);
            LOGGER.info("JSON parsing successful");
            
            // Test data validation
            boolean isValid = processor.validateData(testData);
            LOGGER.info("Data validation result: " + isValid);
            
            LOGGER.info("DataProcessor service started successfully");
            
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error during service initialization", e);
            System.exit(1);
        }
    }
}