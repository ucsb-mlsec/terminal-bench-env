#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <nvToolsExt.h>

class DataLoader {
private:
    std::vector<float> data;
    std::string filename;
    
public:
    DataLoader(const std::string& fname) : filename(fname) {}
    
    bool loadCSVData() {
        nvtxRangePush("Load Input Data");
        
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            nvtxRangePop();
            return false;
        }
        
        std::string line;
        std::vector<std::string> lines;
        
        while (std::getline(file, line)) {
            lines.push_back(line);
        }
        
        file.close();
        nvtxRangePop();
        
        nvtxRangePush("Parse CSV");
        
        for (const auto& l : lines) {
            std::stringstream ss(l);
            std::string value;
            
            while (std::getline(ss, value, ',')) {
                try {
                    float num = std::stof(value);
                    data.push_back(num);
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Could not parse value: " << value << std::endl;
                }
            }
        }
        
        nvtxRangePop();
        
        return true;
    }
    
    const std::vector<float>& getData() const {
        return data;
    }
    
    size_t getDataSize() const {
        return data.size();
    }
    
    void printStats() {
        std::cout << "Loaded " << data.size() << " values from " << filename << std::endl;
        
        if (data.empty()) return;
        
        float sum = 0.0f;
        for (float val : data) {
            sum += val;
        }
        float avg = sum / data.size();
        
        std::cout << "Average value: " << avg << std::endl;
    }
};

int main() {
    DataLoader loader("input_data.csv");
    
    if (loader.loadCSVData()) {
        loader.printStats();
        std::cout << "Data loading completed successfully" << std::endl;
    } else {
        std::cerr << "Failed to load data" << std::endl;
        return 1;
    }
    
    return 0;
}