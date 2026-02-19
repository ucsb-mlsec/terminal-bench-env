#ifndef PARSER_H
#define PARSER_H

#include "validator.h"
#include <string>
#include <vector>

namespace utils {

class Parser {
public:
    Parser();
    ~Parser();
    
    // Parse input string and return result
    bool parse(const std::string& input);
    
    // Tokenize input into components
    std::vector<std::string> tokenize(const std::string& input);
    
    // Set delimiter for tokenization
    void setDelimiter(char delimiter);
    
    // Get last error message
    std::string getLastError() const;
    
private:
    Validator validator_;
    char delimiter_;
    std::string lastError_;
    
    bool validateToken(const std::string& token);
};

} // namespace utils

#endif // PARSER_H