#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>

using namespace std;

// Global configuration pointer
struct Config {
    int bufferSize;
    string mode;
    vector<string> options;
};

Config* globalConfig = nullptr;

// Forward declarations
void loadConfig(const string& configFile);
void processFile(const string& filename);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_file>" << endl;
        return 1;
    }
    
    string inputFile = argv[1];
    
    // Load configuration once at startup
    loadConfig("config.txt");
    
    cout << "Processing file: " << inputFile << endl;
    
    // Process the input file
    processFile(inputFile);
    
    cout << "Processing complete." << endl;
    
    return 0;
}

void loadConfig(const string& configFile) {
    // Allocate configuration structure
    globalConfig = new Config();
    
    globalConfig->bufferSize = 1048576; // 1MB default
    globalConfig->mode = "batch";
    globalConfig->options.push_back("trace");
    globalConfig->options.push_back("profile");
    
    cout << "Configuration loaded (buffer size: " 
         << globalConfig->bufferSize << " bytes)" << endl;
    
    // Memory leak: never delete globalConfig
}

void processFile(const string& filename) {
    ifstream inFile(filename);
    
    if (!inFile.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return;
    }
    
    // Allocate large working buffer for processing
    // This simulates memory-intensive operations
    char* workBuffer = new char[1024 * 1024]; // 1MB buffer
    memset(workBuffer, 0, 1024 * 1024);
    
    // Allocate another buffer for intermediate results
    char* resultBuffer = new char[512 * 1024]; // 512KB buffer
    memset(resultBuffer, 0, 512 * 1024);
    
    string line;
    int lineCount = 0;
    int totalBytes = 0;
    
    cout << "Reading and processing data..." << endl;
    
    while (getline(inFile, line)) {
        lineCount++;
        totalBytes += line.length();
        
        // Simulate processing by copying data to buffers
        if (line.length() > 0) {
            size_t copyLen = min(line.length(), (size_t)1024);
            memcpy(workBuffer + (lineCount % 1000) * 1024, line.c_str(), copyLen);
        }
        
        // Some dummy computation
        for (size_t i = 0; i < line.length() && i < 100; i++) {
            resultBuffer[i] = line[i] ^ 0x55;
        }
    }
    
    inFile.close();
    
    cout << "Processed " << lineCount << " lines (" 
         << totalBytes << " bytes total)" << endl;
    
    // MEMORY LEAK: Forgot to delete the allocated buffers!
    // Should have: delete[] workBuffer;
    // Should have: delete[] resultBuffer;
}