#ifndef CONFIG_H
#define CONFIG_H

// Configuration class for application settings
// Note: std::string is available through precompiled header

class Config {
public:
    std::string appName;
    std::string version;
    std::string serverAddress;
    int maxConnections;
    
    Config() : appName("DefaultApp"), 
               version("1.0.0"), 
               serverAddress("localhost"),
               maxConnections(100) {}
    
    void setAppName(const std::string& name) { appName = name; }
    void setVersion(const std::string& ver) { version = ver; }
    void setServerAddress(const std::string& addr) { serverAddress = addr; }
    void setMaxConnections(int max) { maxConnections = max; }
};

#endif