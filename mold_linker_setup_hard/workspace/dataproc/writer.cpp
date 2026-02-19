#include "writer.h"
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace dataproc {

// Buffer size for write operations
const size_t WRITE_BUFFER_SIZE = 8192;

Writer::Writer() : output_format_(OutputFormat::CSV), buffer_size_(0) {
    write_buffer_.reserve(WRITE_BUFFER_SIZE);
}

Writer::~Writer() {
    try {
        close();
    } catch (...) {
        // Suppress exceptions in destructor
    }
}

bool Writer::open(const std::string& filename, OutputFormat format) {
    if (output_stream_.is_open()) {
        close();
    }
    
    output_format_ = format;
    output_filename_ = filename;
    
    output_stream_.open(filename, std::ios::out | std::ios::trunc);
    if (!output_stream_.is_open()) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return false;
    }
    
    buffer_size_ = 0;
    write_buffer_.clear();
    return true;
}

void Writer::close() {
    if (output_stream_.is_open()) {
        flush();
        output_stream_.close();
    }
}

void Writer::flush() {
    if (buffer_size_ > 0 && output_stream_.is_open()) {
        output_stream_ << write_buffer_;
        output_stream_.flush();
        write_buffer_.clear();
        buffer_size_ = 0;
    }
}

std::string Writer::escapeCSVField(const std::string& field) {
    bool needs_quoting = false;
    
    // Check if field contains special characters
    if (field.find(',') != std::string::npos ||
        field.find('"') != std::string::npos ||
        field.find('\n') != std::string::npos ||
        field.find('\r') != std::string::npos) {
        needs_quoting = true;
    }
    
    if (!needs_quoting) {
        return field;
    }
    
    // Escape quotes by doubling them
    std::string escaped;
    escaped.reserve(field.size() + 10);
    escaped += '"';
    
    for (char c : field) {
        if (c == '"') {
            escaped += "\"\"";
        } else {
            escaped += c;
        }
    }
    
    escaped += '"';
    return escaped;
}

void Writer::writeToBuffer(const std::string& data) {
    write_buffer_ += data;
    buffer_size_ += data.size();
    
    if (buffer_size_ >= WRITE_BUFFER_SIZE) {
        flush();
    }
}

bool Writer::writeRow(const std::vector<std::string>& row) {
    if (!output_stream_.is_open()) {
        std::cerr << "Output stream is not open" << std::endl;
        return false;
    }
    
    if (row.empty()) {
        return true;
    }
    
    std::string line;
    
    switch (output_format_) {
        case OutputFormat::CSV:
            for (size_t i = 0; i < row.size(); ++i) {
                if (i > 0) line += ",";
                line += escapeCSVField(row[i]);
            }
            break;
            
        case OutputFormat::TAB:
            for (size_t i = 0; i < row.size(); ++i) {
                if (i > 0) line += "\t";
                line += row[i];
            }
            break;
            
        case OutputFormat::FIXED:
            for (const auto& field : row) {
                line += std::string(field).substr(0, 20);
                if (field.size() < 20) {
                    line += std::string(20 - field.size(), ' ');
                }
            }
            break;
    }
    
    line += "\n";
    writeToBuffer(line);
    
    return true;
}

bool Writer::writeHeader(const std::vector<std::string>& headers) {
    return writeRow(headers);
}

bool Writer::writeData(const std::vector<std::vector<std::string>>& data) {
    for (const auto& row : data) {
        if (!writeRow(row)) {
            return false;
        }
    }
    return true;
}

bool Writer::writeSummary(const std::string& filename, const SummaryStats& stats) {
    std::ofstream summary_file(filename);
    if (!summary_file.is_open()) {
        std::cerr << "Failed to open summary file: " << filename << std::endl;
        return false;
    }
    
    summary_file << "Data Processing Summary\n";
    summary_file << "=======================\n\n";
    summary_file << "Total Records: " << stats.total_records << "\n";
    summary_file << "Valid Records: " << stats.valid_records << "\n";
    summary_file << "Invalid Records: " << stats.invalid_records << "\n";
    summary_file << "Processing Time: " << std::fixed << std::setprecision(2) 
                 << stats.processing_time_ms << " ms\n";
    
    if (stats.total_records > 0) {
        double valid_percentage = (static_cast<double>(stats.valid_records) / 
                                  stats.total_records) * 100.0;
        summary_file << "Valid Percentage: " << std::fixed << std::setprecision(2)
                    << valid_percentage << "%\n";
    }
    
    summary_file << "\nField Statistics:\n";
    summary_file << "-----------------\n";
    for (const auto& field_stat : stats.field_stats) {
        summary_file << "Field: " << field_stat.field_name << "\n";
        summary_file << "  Non-empty: " << field_stat.non_empty_count << "\n";
        summary_file << "  Empty: " << field_stat.empty_count << "\n";
        summary_file << "  Unique values: " << field_stat.unique_values << "\n";
        summary_file << "\n";
    }
    
    summary_file.close();
    return true;
}

bool Writer::writeErrorLog(const std::string& filename, 
                          const std::vector<std::string>& errors) {
    std::ofstream error_file(filename);
    if (!error_file.is_open()) {
        std::cerr << "Failed to open error log: " << filename << std::endl;
        return false;
    }
    
    error_file << "Error Log\n";
    error_file << "=========\n\n";
    error_file << "Total Errors: " << errors.size() << "\n\n";
    
    for (size_t i = 0; i < errors.size(); ++i) {
        error_file << "[" << (i + 1) << "] " << errors[i] << "\n";
    }
    
    error_file.close();
    return true;
}

void Writer::setOutputFormat(OutputFormat format) {
    output_format_ = format;
}

OutputFormat Writer::getOutputFormat() const {
    return output_format_;
}

bool Writer::isOpen() const {
    return output_stream_.is_open();
}

std::string Writer::getFilename() const {
    return output_filename_;
}

size_t Writer::getBufferSize() const {
    return buffer_size_;
}

} // namespace dataproc