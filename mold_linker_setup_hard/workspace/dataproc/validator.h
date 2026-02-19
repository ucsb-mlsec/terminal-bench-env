#ifndef VALIDATOR_H
#define VALIDATOR_H

#include <string>
#include <vector>
#include <set>

namespace dataproc {

enum class ValidationRuleType {
    NOT_NULL,
    NUMERIC,
    DATE_FORMAT,
    EMAIL_FORMAT,
    RANGE_CHECK,
    UNIQUE_VALUE,
    CUSTOM_REGEX
};

struct ValidationError {
    int row_number;
    int column_index;
    std::string column_name;
    ValidationRuleType rule_type;
    std::string error_message;
};

struct ValidationResult {
    bool is_valid;
    std::vector<ValidationError> errors;
    int total_rows_validated;
    int errors_count;
};

class Validator {
public:
    Validator();
    ~Validator();
    
    ValidationResult validate_csv_data(const std::vector<std::vector<std::string>>& data);
    void add_rule(int column_index, ValidationRuleType rule_type, const std::string& param = "");
    bool validate_cell(const std::string& value, ValidationRuleType rule_type);
    bool is_numeric(const std::string& value);
    bool is_valid_date(const std::string& value);
    bool is_valid_email(const std::string& value);
    void clear_rules();
    
private:
    std::vector<std::pair<int, ValidationRuleType>> validation_rules;
    std::set<std::string> unique_values_cache;
};

} // namespace dataproc

#endif // VALIDATOR_H