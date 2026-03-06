#include "validator.h"
#include <string>
#include <vector>
#include <regex>
#include <set>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <cctype>
#include <cmath>

namespace dataproc {

// Regular expression patterns for validation
static const std::regex EMAIL_PATTERN(
    R"(^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$)"
);

static const std::regex DATE_PATTERN_ISO(
    R"(^\d{4}-\d{2}-\d{2}$)"
);

static const std::regex DATE_PATTERN_US(
    R"(^\d{2}/\d{2}/\d{4}$)"
);

static const std::regex NUMBER_PATTERN(
    R"(^-?\d+\.?\d*$)"
);

static const std::regex INTEGER_PATTERN(
    R"(^-?\d+$)"
);

// Validation error implementation
ValidationError::ValidationError(int line, const std::string& field, 
                                 const std::string& message)
    : line_number(line), field_name(field), error_message(message) {}

std::string ValidationError::to_string() const {
    std::ostringstream oss;
    oss << "Line " << line_number << ", Field '" << field_name 
        << "': " << error_message;
    return oss.str();
}

// Validator implementation
Validator::Validator() : strict_mode(false), max_errors(100) {}

void Validator::set_strict_mode(bool strict) {
    strict_mode = strict;
}

void Validator::set_max_errors(int max) {
    max_errors = max;
}

bool Validator::is_number(const std::string& value) const {
    if (value.empty()) return false;
    return std::regex_match(value, NUMBER_PATTERN);
}

bool Validator::is_integer(const std::string& value) const {
    if (value.empty()) return false;
    return std::regex_match(value, INTEGER_PATTERN);
}

bool Validator::is_date(const std::string& value) const {
    if (value.empty()) return false;
    
    // Check ISO format (YYYY-MM-DD)
    if (std::regex_match(value, DATE_PATTERN_ISO)) {
        return validate_date_components(value, '-');
    }
    
    // Check US format (MM/DD/YYYY)
    if (std::regex_match(value, DATE_PATTERN_US)) {
        return validate_date_components_us(value);
    }
    
    return false;
}

bool Validator::validate_date_components(const std::string& date, char sep) const {
    std::istringstream iss(date);
    std::string year_str, month_str, day_str;
    
    std::getline(iss, year_str, sep);
    std::getline(iss, month_str, sep);
    std::getline(iss, day_str, sep);
    
    int year = std::stoi(year_str);
    int month = std::stoi(month_str);
    int day = std::stoi(day_str);
    
    if (month < 1 || month > 12) return false;
    if (day < 1 || day > 31) return false;
    if (year < 1900 || year > 2100) return false;
    
    // Check days in month
    const int days_in_month[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    int max_day = days_in_month[month - 1];
    
    // Leap year check
    if (month == 2 && is_leap_year(year)) {
        max_day = 29;
    }
    
    return day <= max_day;
}

bool Validator::validate_date_components_us(const std::string& date) const {
    std::istringstream iss(date);
    std::string month_str, day_str, year_str;
    
    std::getline(iss, month_str, '/');
    std::getline(iss, day_str, '/');
    std::getline(iss, year_str, '/');
    
    int month = std::stoi(month_str);
    int day = std::stoi(day_str);
    int year = std::stoi(year_str);
    
    if (month < 1 || month > 12) return false;
    if (day < 1 || day > 31) return false;
    if (year < 1900 || year > 2100) return false;
    
    const int days_in_month[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    int max_day = days_in_month[month - 1];
    
    if (month == 2 && is_leap_year(year)) {
        max_day = 29;
    }
    
    return day <= max_day;
}

bool Validator::is_leap_year(int year) const {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

bool Validator::is_email(const std::string& value) const {
    if (value.empty()) return false;
    if (value.length() > 254) return false; // RFC 5321
    return std::regex_match(value, EMAIL_PATTERN);
}

bool Validator::is_in_range(double value, double min, double max) const {
    return value >= min && value <= max;
}

bool Validator::is_in_set(const std::string& value, 
                          const std::set<std::string>& valid_values) const {
    return valid_values.find(value) != valid_values.end();
}

void Validator::add_required_field(const std::string& field_name) {
    required_fields.insert(field_name);
}

void Validator::add_validation_rule(const ValidationRule& rule) {
    validation_rules.push_back(rule);
}

std::vector<ValidationError> Validator::validate_record(
    int line_number,
    const std::map<std::string, std::string>& record) const {
    
    std::vector<ValidationError> errors;
    
    // Check required fields
    for (const auto& field : required_fields) {
        auto it = record.find(field);
        if (it == record.end() || it->second.empty()) {
            errors.emplace_back(line_number, field, "Required field is missing or empty");
            if (errors.size() >= static_cast<size_t>(max_errors)) {
                return errors;
            }
        }
    }
    
    // Apply validation rules
    for (const auto& rule : validation_rules) {
        auto it = record.find(rule.field_name);
        if (it == record.end()) {
            if (strict_mode) {
                errors.emplace_back(line_number, rule.field_name, 
                                  "Field not found in record");
                if (errors.size() >= static_cast<size_t>(max_errors)) {
                    return errors;
                }
            }
            continue;
        }
        
        const std::string& value = it->second;
        
        // Skip validation for empty values unless required
        if (value.empty() && required_fields.find(rule.field_name) == required_fields.end()) {
            continue;
        }
        
        bool valid = true;
        std::string error_msg;
        
        switch (rule.type) {
            case ValidationType::NUMBER:
                if (!is_number(value)) {
                    valid = false;
                    error_msg = "Value is not a valid number";
                }
                break;
                
            case ValidationType::INTEGER:
                if (!is_integer(value)) {
                    valid = false;
                    error_msg = "Value is not a valid integer";
                }
                break;
                
            case ValidationType::DATE:
                if (!is_date(value)) {
                    valid = false;
                    error_msg = "Value is not a valid date";
                }
                break;
                
            case ValidationType::EMAIL:
                if (!is_email(value)) {
                    valid = false;
                    error_msg = "Value is not a valid email address";
                }
                break;
                
            case ValidationType::RANGE:
                if (is_number(value)) {
                    double num_value = std::stod(value);
                    if (!is_in_range(num_value, rule.min_value, rule.max_value)) {
                        valid = false;
                        error_msg = "Value out of range [" + 
                                  std::to_string(rule.min_value) + ", " + 
                                  std::to_string(rule.max_value) + "]";
                    }
                } else {
                    valid = false;
                    error_msg = "Value must be numeric for range validation";
                }
                break;
                
            case ValidationType::SET:
                if (!is_in_set(value, rule.valid_values)) {
                    valid = false;
                    error_msg = "Value not in allowed set of values";
                }
                break;
                
            case ValidationType::PATTERN:
                if (!rule.pattern.empty()) {
                    try {
                        std::regex pattern(rule.pattern);
                        if (!std::regex_match(value, pattern)) {
                            valid = false;
                            error_msg = "Value does not match required pattern";
                        }
                    } catch (const std::regex_error& e) {
                        valid = false;
                        error_msg = "Invalid regex pattern in rule";
                    }
                }
                break;
        }
        
        if (!valid) {
            errors.emplace_back(line_number, rule.field_name, error_msg);
            if (errors.size() >= static_cast<size_t>(max_errors)) {
                return errors;
            }
        }
    }
    
    return errors;
}

std::vector<ValidationError> Validator::validate_records(
    const std::vector<std::map<std::string, std::string>>& records) const {
    
    std::vector<ValidationError> all_errors;
    int line_number = 1;
    
    for (const auto& record : records) {
        auto errors = validate_record(line_number, record);
        all_errors.insert(all_errors.end(), errors.begin(), errors.end());
        
        if (all_errors.size() >= static_cast<size_t>(max_errors)) {
            break;
        }
        
        line_number++;
    }
    
    return all_errors;
}

void Validator::print_errors(const std::vector<ValidationError>& errors) const {
    std::cout << "Validation Errors (" << errors.size() << " found):" << std::endl;
    for (const auto& error : errors) {
        std::cout << "  " << error.to_string() << std::endl;
    }
}

} // namespace dataproc