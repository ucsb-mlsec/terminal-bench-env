#ifndef DATABASE_H
#define DATABASE_H

#include "connection.h"
#include "logger.h"

class Database {
public:
    Database();
    ~Database();
    
    bool connect(const char* host, int port);
    void disconnect();
    
    bool query(const char* sql);
    bool execute(const char* command);
    
    bool beginTransaction();
    bool commitTransaction();
    bool rollbackTransaction();
    
    int getLastError() const;
    const char* getErrorMessage() const;
    
private:
    Connection* conn;
    Logger* log;
    bool connected;
    int lastError;
};

#endif