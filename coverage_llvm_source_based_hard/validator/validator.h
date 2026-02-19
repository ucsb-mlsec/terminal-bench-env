#ifndef VALIDATOR_H
#define VALIDATOR_H

#include <string>

bool validateEmail(const std::string& email);
bool validatePhoneNumber(const std::string& phone);
bool validateDate(const std::string& date);
bool validateNumericRange(int value, int min, int max);
bool validateURL(const std::string& url);
bool validateCreditCard(const std::string& cardNumber);

#endif