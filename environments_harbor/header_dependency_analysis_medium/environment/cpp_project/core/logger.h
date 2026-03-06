#ifndef LOGGER_H
#define LOGGER_H

#include "utils/formatter.h"
#include <string>

namespace core {

class Logger {
public:
    Logger();
    explicit Logger(const std::string& name);
    ~Logger();

    void log(const std::string& message);
    void error(const std::string& message);
    void debug(const std::string& message);
    void warning(const std::string& message);
    void info(const std::string& message);

    void setLevel(int level);
    int getLevel() const;

private:
    std::string m_name;
    int m_level;
    utils::Formatter m_formatter;

    void writeLog(const std::string& level, const std::string& message);
};

} // namespace core

#endif // LOGGER_H