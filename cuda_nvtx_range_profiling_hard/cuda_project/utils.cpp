#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <ctime>

using namespace std;

// Utility function to print array elements
void printArray(const vector<double>& arr, const string& label) {
    cout << label << ": ";
    for (size_t i = 0; i < arr.size() && i < 10; ++i) {
        cout << fixed << setprecision(4) << arr[i] << " ";
    }
    if (arr.size() > 10) {
        cout << "... (" << arr.size() << " total elements)";
    }
    cout << endl;
}

// Format output with padding and alignment
string formatOutput(const string& key, double value, int width) {
    ostringstream oss;
    oss << left << setw(width) << key << ": " 
        << fixed << setprecision(6) << value;
    return oss.str();
}

// Get current timestamp as a formatted string
string getCurrentTimestamp() {
    auto now = chrono::system_clock::now();
    auto time = chrono::system_clock::to_time_t(now);
    char buffer[100];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", localtime(&time));
    return string(buffer);
}

// Simple string tokenizer
vector<string> tokenize(const string& str, char delimiter) {
    vector<string> tokens;
    stringstream ss(str);
    string token;
    while (getline(ss, token, delimiter)) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

// Log message with timestamp
void logMessage(const string& message, bool verbose) {
    if (verbose) {
        cout << "[" << getCurrentTimestamp() << "] " << message << endl;
    }
}

// Calculate statistics for a dataset
void calculateStats(const vector<double>& data, double& mean, double& min, double& max) {
    if (data.empty()) {
        mean = min = max = 0.0;
        return;
    }
    
    double sum = 0.0;
    min = data[0];
    max = data[0];
    
    for (double val : data) {
        sum += val;
        if (val < min) min = val;
        if (val > max) max = val;
    }
    
    mean = sum / data.size();
}