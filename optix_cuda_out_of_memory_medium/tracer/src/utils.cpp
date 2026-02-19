#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include "utils.h"

using namespace std;

vector<string> readLines(const string& filename) {
    vector<string> lines;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return lines;
    }
    
    string line;
    while (getline(file, line)) {
        lines.push_back(line);
    }
    
    file.close();
    return lines;
}

vector<double> parseNumbers(const string& line) {
    vector<double> numbers;
    stringstream ss(line);
    string token;
    
    while (getline(ss, token, ',')) {
        token = trim(token);
        if (!token.empty()) {
            try {
                double num = stod(token);
                numbers.push_back(num);
            } catch (const exception& e) {
                cerr << "Warning: Could not parse number from '" << token << "'" << endl;
            }
        }
    }
    
    return numbers;
}

string trim(const string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == string::npos) {
        return "";
    }
    
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, (last - first + 1));
}

bool fileExists(const string& path) {
    ifstream file(path);
    bool exists = file.good();
    file.close();
    return exists;
}

void printVector(const vector<double>& vec) {
    cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        cout << vec[i];
        if (i < vec.size() - 1) {
            cout << ", ";
        }
    }
    cout << "]" << endl;
}