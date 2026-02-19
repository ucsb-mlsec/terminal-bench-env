package com.example;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import java.io.StringReader;
import java.io.IOException;

public class CsvProcessor {
    private static final Logger logger = LoggerFactory.getLogger(CsvProcessor.class);

    public static void main(String[] args) {
        logger.info("Starting CSV processor");
        
        String csvData = "name,age\n" +
                        "John Doe,30\n" +
                        "Jane Smith,25\n" +
                        "Bob Johnson,35";
        
        try {
            CSVParser parser = CSVParser.parse(new StringReader(csvData), 
                                              CSVFormat.DEFAULT.withFirstRecordAsHeader());
            
            for (CSVRecord record : parser) {
                logger.info("Processing record: name={}, age={}", 
                           record.get("name"), record.get("age"));
            }
            
            logger.info("CSV processing completed successfully");
        } catch (IOException e) {
            logger.error("Error processing CSV data", e);
        }
    }
}