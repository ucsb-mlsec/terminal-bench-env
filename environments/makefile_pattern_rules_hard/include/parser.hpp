#ifndef PARSER_HPP
#define PARSER_HPP

#include <string>
#include <vector>

// Parse input data and return success status
bool parseData();

// Tokenize input string into vector of tokens
std::vector<std::string> tokenize(const std::string& input);

#endif // PARSER_HPP