#ifndef PARSER_H
#define PARSER_H

#include <string>
#include <vector>
#include <fstream>

// CSV parsing constants
const char CSV_DELIMITER = ',';
const char CSV_QUOTE = '"';
const int MAX_FIELD_LENGTH = 4096;

// Structure to hold a single CSV row
struct CSVRow {
    std::vector<std::string> fields;
    int row_number;
    
    CSVRow() : row_number(0) {}
    size_t size() const { return fields.size(); }
    const std::string& operator[](size_t index) const { return fields[index]; }
};

// Structure to hold parsed CSV data
struct CSVData {
    std::vector<CSVRow> rows;
    std::vector<std::string> headers;
    size_t column_count;
    
    CSVData() : column_count(0) {}
};

// Function declarations
CSVData* parse_csv_file(const std::string& filename, bool has_header = true);
bool parse_csv_line(const std::string& line, CSVRow& row);
std::string trim_whitespace(const std::string& str);
bool validate_csv_row(const CSVRow& row, size_t expected_columns);
void free_csv_data(CSVData* data);

#endif // PARSER_H