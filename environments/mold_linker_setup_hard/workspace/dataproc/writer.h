#ifndef WRITER_H
#define WRITER_H

#include <string>
#include <vector>
#include <fstream>
#include <memory>

namespace dataproc {

// Output format types
enum class OutputFormat {
    CSV,
    JSON,
    XML,
    BINARY
};

// Forward declarations
class DataRow;

// Writer base class for output operations
class Writer {
public:
    Writer(const std::string& filename, OutputFormat format);
    virtual ~Writer();
    
    bool open();
    void close();
    bool isOpen() const;
    
    virtual bool writeHeader(const std::vector<std::string>& columns) = 0;
    virtual bool writeRow(const DataRow& row) = 0;
    virtual bool writeRows(const std::vector<DataRow>& rows) = 0;
    
protected:
    std::string filename_;
    OutputFormat format_;
    std::ofstream output_;
    bool isOpen_;
};

// Factory function for creating writers
std::unique_ptr<Writer> createWriter(const std::string& filename, OutputFormat format);

// Buffered writer for improved performance
class BufferedWriter : public Writer {
public:
    BufferedWriter(const std::string& filename, OutputFormat format, size_t bufferSize = 8192);
    
    bool writeHeader(const std::vector<std::string>& columns) override;
    bool writeRow(const DataRow& row) override;
    bool writeRows(const std::vector<DataRow>& rows) override;
    void flush();
    
private:
    std::vector<char> buffer_;
    size_t bufferPos_;
};

} // namespace dataproc

#endif // WRITER_H