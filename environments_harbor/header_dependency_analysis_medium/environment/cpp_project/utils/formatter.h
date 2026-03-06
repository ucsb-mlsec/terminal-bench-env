#ifndef FORMATTER_H
#define FORMATTER_H

#include "../utils/parser.h"

namespace Utils {

class Formatter {
public:
    Formatter();
    ~Formatter();
    
    // Format input string according to specified rules
    std::string format(const std::string& input);
    
    // Format with custom delimiter
    std::string format(const std::string& input, char delimiter);
    
    // Convert internal state to string representation
    std::string toString() const;
    
    // Set formatting options
    void setIndentation(int spaces);
    void setMaxLineLength(int length);
    
    // Parse and format in one operation
    std::string parseAndFormat(const std::string& raw);
    
private:
    Parser parser_;
    int indentation_;
    int maxLineLength_;
    
    std::string applyFormatting(const std::string& text);
};

} // namespace Utils

#endif