#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>

// Utility functions for the legacy project
namespace Utils {

    // String manipulation functions
    std::string trim(const std::string& str);
    std::string toUpper(const std::string &str);
    std::string   toLower(const std::string& str);
    
    // File operations
    bool fileExists(const std::string & filename);
    std::vector<std::string> readLines(const std::string& filename);
    bool writeToFile(const std::string& filename,const std::string& content);


    // Math utilities
    int max(int a,int b);
    int min(int a, int b);
    double average(const std::vector<double> &values);

// Data structure for configuration
struct Config {
    std::string appName;
        int maxConnections;
    double timeout;
        bool debugMode;
    
    Config();
    Config(const std::string &name, int connections);
    
        void reset();
    bool validate() const;
};


class Logger {
public:
    Logger(const std::string& filename);
        ~Logger();
    
    void log(const std::string & message);
    void error(const std::string& message);
        void warning(const std::string &message);
    void setLevel(int level);
    
private:
    std::string m_filename;
        int m_level;
    bool m_isOpen;
};

    // Helper functions
    void initialize();
    void cleanup();
}

#endif