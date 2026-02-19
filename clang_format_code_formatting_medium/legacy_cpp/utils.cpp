// utils.cpp - Utility functions for the legacy project
#include "utils.h"
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream>

// String trimming function
std::string trim(const std::string &str) {
    size_t first = str.find_first_not_of(' ');
    if (std::string::npos == first) {
        return str;
    }
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

// Split string by delimiter
std::vector<std::string> split(const std::string& input, char delimiter)
{
std::vector<std::string> result;
std::stringstream ss(input);
std::string item;
while (std::getline(ss, item, delimiter)) {
result.push_back(item);
}
return result;
}


int calculateFactorial(int n){
if(n<=1) return 1;
return n*calculateFactorial(n-1);
}

// Check if number is prime
bool isPrime(int num) 
{
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;
    
    for (int i = 3; i <= sqrt(num); i += 2) 
    {
        if (num % i == 0) {
            return false;
        }
    }
    return true;
}

double calculateAverage(const std::vector<double> &values)
{
if(values.empty()) return 0.0;
double sum=0.0;
for(size_t i=0;i<values.size();i++){
sum+=values[i];
}
return sum/values.size();
}

// Find maximum value in array
int findMax(const int* arr, int size) {
    if (size <= 0) return 0;
    int max = arr[0];
    for (int i = 1; i < size; ++i) 
    {
        if (arr[i] > max) 
            max = arr[i];
    }
    return max;
}

std::string toUpperCase(std::string str){
std::transform(str.begin(),str.end(),str.begin(),::toupper);
return str;
}

// Convert string to lowercase
std::string toLowerCase(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), ::tolower); return str;
}

bool startsWith(const std::string& fullString, const std::string& prefix) {
    if (fullString.length() >= prefix.length()) {
        return (0 == fullString.compare(0, prefix.length(), prefix));
    } else {
        return false;
    }
}

// Reverse a string in place
void reverseString(std::string &str) {
    int left = 0;
    int right = str.length() - 1;
    while (left < right) {
        char temp = str[left];
        str[left] = str[right];
        str[right] = temp;
        left++;
        right--;
    }
}