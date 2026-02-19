#include <string>
#include <regex>
#include <cctype>
#include <algorithm>
#include <vector>

bool validateEmail(const std::string& email) {
    if (email.empty() || email.length() > 254) {
        return false;
    }
    
    size_t atPos = email.find('@');
    if (atPos == std::string::npos || atPos == 0) {
        return false;
    }
    
    std::string localPart = email.substr(0, atPos);
    std::string domainPart = email.substr(atPos + 1);
    
    if (localPart.empty() || localPart.length() > 64) {
        return false;
    }
    
    if (domainPart.empty() || domainPart.find('.') == std::string::npos) {
        return false;
    }
    
    if (localPart[0] == '.' || localPart[localPart.length() - 1] == '.') {
        return false;
    }
    
    if (domainPart[0] == '.' || domainPart[domainPart.length() - 1] == '.') {
        return false;
    }
    
    std::regex emailPattern(R"([a-zA-Z0-9._%-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})");
    return std::regex_match(email, emailPattern);
}

bool validatePhoneNumber(const std::string& phone) {
    std::string digits;
    for (char c : phone) {
        if (std::isdigit(c)) {
            digits += c;
        }
    }
    
    if (digits.length() < 10 || digits.length() > 15) {
        return false;
    }
    
    if (digits.length() == 10) {
        if (digits[0] == '0' || digits[0] == '1') {
            return false;
        }
    }
    
    if (digits.length() == 11 && digits[0] != '1') {
        return false;
    }
    
    return true;
}

bool validateDate(const std::string& date) {
    if (date.length() != 10) {
        return false;
    }
    
    if (date[4] != '-' || date[7] != '-') {
        return false;
    }
    
    std::string yearStr = date.substr(0, 4);
    std::string monthStr = date.substr(5, 2);
    std::string dayStr = date.substr(8, 2);
    
    for (char c : yearStr) {
        if (!std::isdigit(c)) return false;
    }
    for (char c : monthStr) {
        if (!std::isdigit(c)) return false;
    }
    for (char c : dayStr) {
        if (!std::isdigit(c)) return false;
    }
    
    int year = std::stoi(yearStr);
    int month = std::stoi(monthStr);
    int day = std::stoi(dayStr);
    
    if (year < 1900 || year > 2100) {
        return false;
    }
    
    if (month < 1 || month > 12) {
        return false;
    }
    
    if (day < 1 || day > 31) {
        return false;
    }
    
    if (month == 2) {
        bool isLeap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
        if (day > (isLeap ? 29 : 28)) {
            return false;
        }
    }
    
    if (month == 4 || month == 6 || month == 9 || month == 11) {
        if (day > 30) {
            return false;
        }
    }
    
    return true;
}

bool validateNumericRange(int value, int min, int max) {
    if (min > max) {
        return false;
    }
    
    if (value < min) {
        return false;
    }
    
    if (value > max) {
        return false;
    }
    
    return true;
}

bool validateURL(const std::string& url) {
    if (url.empty() || url.length() > 2048) {
        return false;
    }
    
    if (url.find("http://") != 0 && url.find("https://") != 0 && url.find("ftp://") != 0) {
        return false;
    }
    
    size_t protocolEnd = url.find("://");
    if (protocolEnd == std::string::npos) {
        return false;
    }
    
    std::string afterProtocol = url.substr(protocolEnd + 3);
    if (afterProtocol.empty()) {
        return false;
    }
    
    size_t domainEnd = afterProtocol.find('/');
    std::string domain = (domainEnd != std::string::npos) ? 
                         afterProtocol.substr(0, domainEnd) : afterProtocol;
    
    if (domain.empty() || domain.find('.') == std::string::npos) {
        return false;
    }
    
    if (domain[0] == '.' || domain[domain.length() - 1] == '.') {
        return false;
    }
    
    for (char c : domain) {
        if (!std::isalnum(c) && c != '.' && c != '-' && c != ':') {
            return false;
        }
    }
    
    size_t colonPos = domain.find(':');
    if (colonPos != std::string::npos) {
        std::string portStr = domain.substr(colonPos + 1);
        for (char c : portStr) {
            if (!std::isdigit(c)) {
                return false;
            }
        }
        int port = std::stoi(portStr);
        if (port < 1 || port > 65535) {
            return false;
        }
    }
    
    return true;
}

bool validateCreditCard(const std::string& cardNumber) {
    std::string digits;
    for (char c : cardNumber) {
        if (std::isdigit(c)) {
            digits += c;
        }
    }
    
    if (digits.length() < 13 || digits.length() > 19) {
        return false;
    }
    
    int sum = 0;
    bool alternate = false;
    
    for (int i = digits.length() - 1; i >= 0; i--) {
        int digit = digits[i] - '0';
        
        if (alternate) {
            digit *= 2;
            if (digit > 9) {
                digit -= 9;
            }
        }
        
        sum += digit;
        alternate = !alternate;
    }
    
    return (sum % 10 == 0);
}