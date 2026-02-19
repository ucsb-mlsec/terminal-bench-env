#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <fstream>

// Log level enumeration
enum LogLevel{
    DEBUG=0,
    INFO = 1,
  WARNING=2,
    ERROR = 3,
  CRITICAL= 4
};

#define LOG_DEBUG(msg) Logger::getInstance().log(DEBUG, msg)
#define LOG_INFO( msg ) Logger::getInstance().log(INFO,msg)
#define LOG_ERROR(msg)   Logger::getInstance().log(ERROR, msg)


class Logger {
public:
    static Logger& getInstance();
    
  void log(LogLevel level,const std::string& message);
    void setLogLevel( LogLevel level );
    LogLevel getLogLevel() const;
    
    void setOutputFile(const std::string &filename);
  void enableConsoleOutput(bool enable);
    
    bool isEnabled(LogLevel level) const;

private:
    Logger();
  ~Logger();
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
  LogLevel currentLevel;
    std::ofstream logFile;
    bool consoleEnabled;
    
    std::string levelToString(LogLevel level)const;
  void writeLog(const std::string& formattedMessage);
};


#endif