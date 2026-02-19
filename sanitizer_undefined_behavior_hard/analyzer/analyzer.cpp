#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

class DataAnalyzer {
private:
    vector<int> data;
    int* statsBuffer;
    int bufferSize;

public:
    DataAnalyzer() {
        bufferSize = 5;
        statsBuffer = new int[bufferSize];
    }

    ~DataAnalyzer() {
        delete[] statsBuffer;
    }

    bool loadData(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Error: Could not open file " << filename << endl;
            return false;
        }

        string line;
        while (getline(file, line)) {
            if (!line.empty()) {
                data.push_back(stoi(line));
            }
        }
        file.close();

        if (data.empty()) {
            cerr << "Error: No data loaded" << endl;
            return false;
        }

        return true;
    }

    int calculateSum() {
        int sum = 0;
        for (size_t i = 0; i < data.size(); i++) {
            sum += data[i];
        }
        return sum;
    }

    double calculateMean() {
        int sum = calculateSum();
        // UB Issue 1: Division by zero if data is empty
        return static_cast<double>(sum) / data.size();
    }

    int calculateRange() {
        if (data.empty()) return 0;
        
        int minVal = data[0];
        int maxVal = data[0];
        
        for (size_t i = 1; i < data.size(); i++) {
            if (data[i] < minVal) minVal = data[i];
            if (data[i] > maxVal) maxVal = data[i];
        }
        
        return maxVal - minVal;
    }

    void computeAdvancedStats() {
        // UB Issue 2: Out-of-bounds array access
        // statsBuffer has size 5, but we access index 10
        for (int i = 0; i <= 10; i++) {
            statsBuffer[i] = i * 2;
        }
    }

    int calculateScaledSum() {
        int scaledSum = 0;
        // UB Issue 3: Signed integer overflow
        // If sum of data is large, multiplying by large factor causes overflow
        for (size_t i = 0; i < data.size(); i++) {
            int scaled = data[i] * 1000000;
            scaledSum += scaled;
        }
        return scaledSum;
    }

    int getModuloChecksum() {
        if (data.empty()) return 0;
        
        int checksum = 0;
        int divisor = data.size() - 10;
        
        // UB Issue 4: Modulo by zero or negative number
        // If data.size() is 10, divisor becomes 0
        for (size_t i = 0; i < data.size(); i++) {
            checksum += data[i] % divisor;
        }
        
        return checksum;
    }

    void printStatistics() {
        cout << "Data Analysis Results:" << endl;
        cout << "=====================" << endl;
        cout << "Number of values: " << data.size() << endl;
        cout << "Sum: " << calculateSum() << endl;
        cout << "Mean: " << calculateMean() << endl;
        cout << "Range: " << calculateRange() << endl;
        
        computeAdvancedStats();
        cout << "Advanced stats computed" << endl;
        
        int modCheck = getModuloChecksum();
        cout << "Modulo checksum: " << modCheck << endl;
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <datafile>" << endl;
        return 1;
    }

    DataAnalyzer analyzer;
    
    if (!analyzer.loadData(argv[1])) {
        return 1;
    }

    analyzer.printStatistics();
    
    cout << "\nFinal Sum: " << analyzer.calculateSum() << endl;

    return 0;
}