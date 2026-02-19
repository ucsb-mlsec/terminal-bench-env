#ifndef CONFIG_H
#define CONFIG_H

#include <string>

// Configuration constants for the legacy application
#define MAX_BUFFER_SIZE 4096
#define DEFAULT_PORT 8080
#define CONNECTION_TIMEOUT 30
#define MAX_CONNECTIONS 100
#define APP_VERSION "1.2.3"
#define LOG_FILE_PATH "/var/log/app.log"

// Default configuration values
#define DEFAULT_THREAD_POOL_SIZE 10
#define MAX_RETRY_ATTEMPTS 3
#define KEEPALIVE_INTERVAL 60

// Configuration data structure
struct ConfigData {
    std::string serverAddress;
    std::string databaseName;
    int port;
    int maxUsers;
    bool enableLogging;
};

// Global configuration object
extern ConfigData globalConfig;

#endif // CONFIG_H