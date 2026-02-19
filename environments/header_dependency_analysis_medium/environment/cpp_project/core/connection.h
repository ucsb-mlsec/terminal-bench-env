#ifndef CONNECTION_H
#define CONNECTION_H

#include "database.h"
#include "config.h"

namespace Core {

class Connection {
public:
    Connection();
    ~Connection();
    
    bool open();
    bool close();
    bool isActive() const;
    
    void setConnectionString(const char* connStr);
    const char* getConnectionString() const;
    
    bool executeQuery(const char* query);
    void setTimeout(int seconds);
    int getTimeout() const;
    
private:
    bool active;
    char* connectionString;
    int timeout;
    Database* dbInstance;
};

} // namespace Core

#endif // CONNECTION_H