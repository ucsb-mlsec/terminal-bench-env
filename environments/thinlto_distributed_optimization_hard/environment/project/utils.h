#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>

void initializeUtils();
std::string formatString(const std::string& input);
int calculateSum(const std::vector<int>& numbers);
void processData(const std::string& data);
bool validateInput(const std::string& input);

#endif