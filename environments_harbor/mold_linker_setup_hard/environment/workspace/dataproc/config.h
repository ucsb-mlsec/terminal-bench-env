#ifndef DATAPROC_CONFIG_H
#define DATAPROC_CONFIG_H

#include <string>
#include <map>
#include <cstddef>

namespace dataproc {

class Config {
public:
    Config();
    ~Config();
    
    // Load configuration from file
    bool loadFromFile(const std::string& filepath);
    
    // Load configuration from command line arguments
    bool loadFromArgs(int argc, char* argv[]);
    
    // Getters for configuration options
    size_t getThreadCount() const { return thread_count_; }
    size_t getBufferSize() const { return buffer_size_; }
    const std::string& getInputPath() const { return input_path_; }
    const std::string& getOutputPath() const { return output_path_; }
    bool getVerbose() const { return verbose_; }
    int getCompressionLevel() const { return compression_level_; }
    
    // Setters for configuration options
    void setThreadCount(size_t count) { thread_count_ = count; }
    void setBufferSize(size_t size) { buffer_size_ = size; }
    void setInputPath(const std::string& path) { input_path_ = path; }
    void setOutputPath(const std::string& path) { output_path_ = path; }
    
private:
    size_t thread_count_;
    size_t buffer_size_;
    std::string input_path_;
    std::string output_path_;
    bool verbose_;
    int compression_level_;
    std::map<std::string, std::string> extra_options_;
    
    void setDefaults();
    bool parseConfigLine(const std::string& line);
};

} // namespace dataproc

#endif // DATAPROC_CONFIG_H