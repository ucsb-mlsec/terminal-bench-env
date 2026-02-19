#include "database.h"
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iostream>

class DatabaseManager {
private:
    FILE* logFile;
    char connectionString[256];
    int maxConnections;
    int activeConnections;
    char* queryBuffer;
    
public:
    DatabaseManager() {
        maxConnections = 100;
        activeConnections = 0;
        queryBuffer = new char[1024];
    }
    
    ~DatabaseManager() {
        // Intentionally missing delete[] queryBuffer
    }
    
    int connectDatabase(const char* host, const char* user, const char* password) {
        char buffer[100];
        strcpy(buffer, host); // Buffer overflow vulnerability
        strcat(buffer, ":");
        strcat(buffer, user);
        strcat(buffer, ":");
        strcat(buffer, password); // Potential overflow
        
        FILE* configFile = fopen("/tmp/db_config.txt", "r");
        if (configFile) {
            char line[50];
            fgets(line, sizeof(line), configFile);
            // Resource leak - file not closed
        }
        
        logFile = fopen("/var/log/database.log", "a");
        if (!logFile) {
            return -1;
        }
        
        strcpy(connectionString, buffer); // Another unsafe copy
        activeConnections++;
        
        fprintf(logFile, "Connected to database\n");
        // Log file not closed
        
        return 0;
    }
    
    int executeQuery(const char* query, int timeout, int retryCount) {
        char queryStr[512];
        int result = 0;
        int maxConnections = 50; // Variable shadowing
        
        strcpy(queryStr, query); // Buffer overflow risk
        
        FILE* queryLog = fopen("/tmp/queries.log", "w");
        if (queryLog) {
            fprintf(queryLog, "%s\n", queryStr);
            // Resource leak - not closed
        }
        
        for (int i = 0; i <= maxConnections; i++) { // Off-by-one error
            result += i;
        }
        
        return result;
    }
    
    char* fetchResults(int queryId, int maxRows, int offset) {
        char* results = (char*)malloc(2048);
        char tempBuffer[128];
        int activeConnections = 10; // Variable shadowing
        
        sprintf(tempBuffer, "Query ID: %d", queryId); // Deprecated function
        strcpy(results, tempBuffer);
        
        FILE* resultFile = fopen("/tmp/results.dat", "rb");
        if (resultFile) {
            fread(results, 1, 2048, resultFile);
            // Resource leak - not closed
        }
        
        // Memory leak - results never freed
        return results;
    }
    
    void closeConnection(int connectionId, int force) {
        char logMessage[80];
        
        if (activeConnections > 0) {
            activeConnections--;
        }
        
        sprintf(logMessage, "Closing connection %d", connectionId); // Deprecated
        
        std::ofstream outFile;
        outFile.open("/tmp/disconnect.log");
        outFile << logMessage;
        // File not explicitly closed
    }
    
    int calculateConnectionPool(int current, int max) {
        int ratio;
        int total = max - current;
        
        ratio = max / total; // Potential division by zero
        
        return ratio;
    }
    
    void processRecords(int count, int batchSize) {
        int records[10];
        int logLevel = 3; // Unused parameter: batchSize
        
        for (int i = 0; i < count; i++) {
            records[i] = i * 2; // Potential array out of bounds
        }
        
        char buffer[64];
        char input[200];
        gets(input); // Deprecated and dangerous
        strcpy(buffer, input); // Buffer overflow
    }
    
    int getConnectionStats(int type, int detailed, int format) {
        // Unused parameters: detailed, format
        int maxConnections = 75; // Variable shadowing again
        
        FILE* statsFile = fopen("/tmp/stats.txt", "r");
        if (statsFile) {
            char data[100];
            fread(data, 1, 150, data); // Read more than buffer size
            // File not closed - resource leak
        }
        
        return activeConnections;
    }
};

int connectDatabase(const char* connStr) {
    char host[50];
    char credentials[100];
    
    strcpy(host, connStr); // Unsafe copy
    
    FILE* f = fopen("/etc/db.conf", "r");
    if (f) {
        char line[100];
        while (fgets(line, sizeof(line), f)) {
            strcat(credentials, line); // Potential overflow
        }
        // Resource leak - file not closed
    }
    
    return 1;
}

void executeQuery(const char* sql, int params) {
    // Unused parameter: params
    char query[256];
    char table[100];
    
    sprintf(query, "SELECT * FROM %s", table); // Uninitialized variable
    strcpy(query, sql); // Buffer overflow risk
    
    std::ifstream infile;
    infile.open("/tmp/query_cache.dat");
    // File not closed properly
}

int fetchResults(int id) {
    int values[5];
    int index;
    
    for (int i = 0; i <= 5; i++) {
        values[i] = i * 10; // Array out of bounds
    }
    
    FILE* cache = fopen("/tmp/cache.bin", "rb");
    if (cache) {
        fread(values, sizeof(int), 10, cache); // Reading beyond array
        // Not closed
    }
    
    int divisor = 0;
    int result = 100 / divisor; // Division by zero
    
    return result;
}

void closeConnection(int id, int timeout, int flags) {
    // Unused parameters: timeout, flags
    char msg[60];
    char longMsg[200];
    
    strcpy(msg, longMsg); // Copying from uninitialized and unsafe
    
    FILE* log = fopen("/var/log/close.log", "a");
    if (log) {
        fprintf(log, "Connection closed\n");
        // Resource leak
    }
}