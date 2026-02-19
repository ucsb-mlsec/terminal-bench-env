#ifndef VALIDATOR_H
#define VALIDATOR_H

#include "formatter.h"
#include <string>
#include <vector>

namespace utils {

class Validator {
public:
    Validator();
    ~Validator();
    
    bool validate(const std::string& input);
    bool validateAll(const std::vector<std::string>& inputs);
    
    bool check(const std::string& data);
    bool checkFormat(const std::string& data);
    bool checkIntegrity(const std::string& data);
    
    void setStrictMode(bool strict);
    bool isValid() const;
    
    std::string getLastError() const;
    
private:
    bool strictMode;
    std::string lastError;
    Formatter formatter;
};

} // namespace utils

#endif // VALIDATOR_H