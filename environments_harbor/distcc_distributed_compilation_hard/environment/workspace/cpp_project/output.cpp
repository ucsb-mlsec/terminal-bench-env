#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <map>

void printResults(const std::vector<double>& results) {
    std::cout << "\n=== Computation Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    
    if (results.empty()) {
        std::cout << "No results to display." << std::endl;
        return;
    }
    
    std::cout << std::setw(10) << "Index" << std::setw(20) << "Value" << std::endl;
    std::cout << std::string(30, '-') << std::endl;
    
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << std::setw(10) << i 
                  << std::setw(20) << results[i] << std::endl;
    }
    
    std::cout << std::string(30, '=') << std::endl;
}

std::string formatReport(const std::map<std::string, double>& data) {
    std::ostringstream report;
    report << std::fixed << std::setprecision(2);
    
    report << "\n+--------------------+------------------+" << std::endl;
    report << "| Metric             | Value            |" << std::endl;
    report << "+--------------------+------------------+" << std::endl;
    
    for (const auto& entry : data) {
        report << "| " << std::left << std::setw(18) << entry.first 
               << " | " << std::right << std::setw(16) << entry.second 
               << " |" << std::endl;
    }
    
    report << "+--------------------+------------------+" << std::endl;
    
    return report.str();
}

void displaySummary(int count, double average, double total) {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << std::setw(30) << "SUMMARY STATISTICS" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::left << std::setw(25) << "Total Items:" 
              << std::right << std::setw(15) << count << std::endl;
    
    std::cout << std::left << std::setw(25) << "Average Value:" 
              << std::right << std::setw(15) << average << std::endl;
    
    std::cout << std::left << std::setw(25) << "Total Sum:" 
              << std::right << std::setw(15) << total << std::endl;
    
    std::cout << std::string(50, '=') << std::endl << std::endl;
}

void printProgressBar(int current, int total, const std::string& label) {
    const int barWidth = 40;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(barWidth * progress);
    
    std::cout << "\r" << label << " [";
    
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    
    std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100.0) << "%";
    std::cout.flush();
    
    if (current == total) {
        std::cout << std::endl;
    }
}