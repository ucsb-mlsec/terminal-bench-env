#ifndef PARSER_H
#define PARSER_H

#include <string>
#include <vector>

bool parseConfig(const std::string& filename);
void parseInputFile();
std::vector<std::string> parseArguments(int argc, char* argv[]);

#endif