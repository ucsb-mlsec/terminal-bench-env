#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <sstream>

class Logger {
private:
    std::ofstream logFile;
    std::string logPath;
	int logLevel;
    
public:
    enum Level {
        DEBUG = 0,
        INFO = 1,
        WARNING = 2,
        ERROR = 3
    };
    
    Logger(const std::string& path,int level): logPath(path), logLevel(level) {
        logFile.open(logPath, std::ios::app);
    }
    
    ~Logger() {
        if (logFile.is_open()) {
            logFile.close();
        }
    }
    
    std::string getCurrentTimestamp()
    {
        time_t now = time(0);
        char* dt = ctime(&now);
        std::string timestamp(dt);
        timestamp.erase(timestamp.length() - 1);
        return timestamp;
    }
    
    void log(Level level,const std::string &message,const std::string &module="GENERAL") {
        if (level < logLevel) return;
        
        std::string levelStr;
        switch(level){
            case DEBUG: levelStr = "DEBUG"; break;
            case INFO: levelStr = "INFO"; break;
            case WARNING: levelStr = "WARNING"; break;
            case ERROR: levelStr = "ERROR"; break;
        }
        
        std::stringstream ss;
        ss << "[" << getCurrentTimestamp() << "] [" << levelStr << "] [" << module << "] " << message << std::endl;
        
        if (logFile.is_open()) {
            logFile << ss.str();
            logFile.flush();
        }
        std::cout << ss.str();
    }
    
    void debug(const std::string& message,const std::string& module = "GENERAL") {
        log(DEBUG,message,module);
    }
    
    void info(const std::string &message, const std::string &module="GENERAL"){
        log(INFO, message, module);
    }
    
    void warning(const std::string* message, const std::string& module = "GENERAL") {
        log(WARNING,*message,module);
    }
    
    void error(const std::string &message,const std::string &module = "GENERAL") {
        log(ERROR, message, module);
    }
};