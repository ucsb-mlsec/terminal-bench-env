#ifndef DATABASE_H
#define DATABASE_H

#include <string>
#include <vector>

enum ConnectionStatus {
    DISCONNECTED = 0,
    CONNECTING = 1,
    CONNECTED = 2,
    ERROR = 3
};

enum QueryType {
    SELECT,
    INSERT,
    UPDATE,
    DELETE
};

typedef std::vector<std::string> ResultRow;
typedef std::vector<ResultRow> ResultSet;

class DatabaseConnection {
public:
    DatabaseConnection();
    DatabaseConnection(const std::string& host, int port);
    ~DatabaseConnection();
    
    bool connect(const std::string& connectionString);
    void disconnect();
    ResultSet query(const std::string& sql);
    ResultRow fetch();
    ConnectionStatus getStatus() const;
    void setConnectionString(const std::string& connStr);
    
private:
    std::string connectionString;
    ConnectionStatus status;
    int port;
};

#endif