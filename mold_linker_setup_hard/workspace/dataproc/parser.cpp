#include "parser.h"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <cctype>

namespace dataproc {

// Helper function to trim whitespace from the left
std::string ltrim(const std::string& str) {
    size_t start = 0;
    while (start < str.length() && std::isspace(static_cast<unsigned char>(str[start]))) {
        start++;
    }
    return str.substr(start);
}

// Helper function to trim whitespace from the right
std::string rtrim(const std::string& str) {
    size_t end = str.length();
    while (end > 0 && std::isspace(static_cast<unsigned char>(str[end - 1]))) {
        end--;
    }
    return str.substr(0, end);
}

// Helper function to trim whitespace from both sides
std::string trim(const std::string& str) {
    return ltrim(rtrim(str));
}

// Constructor
CSVParser::CSVParser() : has_header_(false), current_line_(0), field_count_(0) {
}

// Destructor
CSVParser::~CSVParser() {
    if (file_stream_.is_open()) {
        file_stream_.close();
    }
}

// Open a CSV file for parsing
bool CSVParser::open(const std::string& filename) {
    filename_ = filename;
    file_stream_.open(filename);
    
    if (!file_stream_.is_open()) {
        last_error_ = "Failed to open file: " + filename;
        return false;
    }
    
    current_line_ = 0;
    return true;
}

// Close the CSV file
void CSVParser::close() {
    if (file_stream_.is_open()) {
        file_stream_.close();
    }
    current_line_ = 0;
}

// Set whether the CSV has a header row
void CSVParser::setHasHeader(bool has_header) {
    has_header_ = has_header;
}

// Parse the header row if present
bool CSVParser::parseHeader() {
    if (!has_header_) {
        return true;
    }
    
    if (!file_stream_.is_open()) {
        last_error_ = "File not open";
        return false;
    }
    
    std::string line;
    if (std::getline(file_stream_, line)) {
        current_line_++;
        header_ = parseLine(line);
        field_count_ = header_.size();
        return true;
    }
    
    last_error_ = "Failed to read header line";
    return false;
}

// Get the header fields
std::vector<std::string> CSVParser::getHeader() const {
    return header_;
}

// Read and parse the next line from the CSV file
bool CSVParser::readNextLine(std::vector<std::string>& fields) {
    if (!file_stream_.is_open()) {
        last_error_ = "File not open";
        return false;
    }
    
    std::string line;
    if (std::getline(file_stream_, line)) {
        current_line_++;
        fields = parseLine(line);
        
        // Validate field count if header was present
        if (has_header_ && field_count_ > 0 && fields.size() != field_count_) {
            std::ostringstream oss;
            oss << "Line " << current_line_ << ": Expected " << field_count_ 
                << " fields, got " << fields.size();
            last_error_ = oss.str();
            return false;
        }
        
        return true;
    }
    
    return false; // End of file
}

// Parse a single CSV line into fields
std::vector<std::string> CSVParser::parseLine(const std::string& line) {
    std::vector<std::string> fields;
    std::string current_field;
    bool in_quotes = false;
    bool escape_next = false;
    
    for (size_t i = 0; i < line.length(); i++) {
        char c = line[i];
        
        if (escape_next) {
            current_field += c;
            escape_next = false;
            continue;
        }
        
        if (c == '\\') {
            escape_next = true;
            continue;
        }
        
        if (c == '"') {
            if (in_quotes) {
                // Check for escaped quote (double quote)
                if (i + 1 < line.length() && line[i + 1] == '"') {
                    current_field += '"';
                    i++; // Skip the next quote
                } else {
                    in_quotes = false;
                }
            } else {
                in_quotes = true;
            }
            continue;
        }
        
        if (c == ',' && !in_quotes) {
            fields.push_back(trim(current_field));
            current_field.clear();
            continue;
        }
        
        current_field += c;
    }
    
    // Add the last field
    fields.push_back(trim(current_field));
    
    return fields;
}

// Handle quoted fields in CSV
std::string CSVParser::handleQuotedField(const std::string& field) {
    if (field.empty()) {
        return field;
    }
    
    std::string result = field;
    
    // Remove surrounding quotes if present
    if (result.front() == '"' && result.back() == '"') {
        result = result.substr(1, result.length() - 2);
    }
    
    // Replace escaped quotes
    size_t pos = 0;
    while ((pos = result.find("\"\"", pos)) != std::string::npos) {
        result.replace(pos, 2, "\"");
        pos++;
    }
    
    return result;
}

// Validate CSV data structure
bool CSVParser::validate() {
    if (!file_stream_.is_open()) {
        last_error_ = "File not open";
        return false;
    }
    
    // Save current position
    std::streampos original_pos = file_stream_.tellg();
    size_t original_line = current_line_;
    
    // Reset to beginning
    file_stream_.clear();
    file_stream_.seekg(0);
    current_line_ = 0;
    
    bool valid = true;
    std::string line;
    size_t expected_fields = 0;
    
    while (std::getline(file_stream_, line)) {
        current_line_++;
        
        if (line.empty()) {
            continue;
        }
        
        std::vector<std::string> fields = parseLine(line);
        
        if (current_line_ == 1) {
            expected_fields = fields.size();
        } else if (fields.size() != expected_fields) {
            std::ostringstream oss;
            oss << "Line " << current_line_ << ": Inconsistent field count";
            last_error_ = oss.str();
            valid = false;
            break;
        }
    }
    
    // Restore original position
    file_stream_.clear();
    file_stream_.seekg(original_pos);
    current_line_ = original_line;
    
    return valid;
}

// Get the last error message
std::string CSVParser::getLastError() const {
    return last_error_;
}

// Get the current line number
size_t CSVParser::getCurrentLine() const {
    return current_line_;
}

// Check if end of file has been reached
bool CSVParser::isEOF() const {
    return file_stream_.eof();
}

} // namespace dataproc