#include <iostream>
#include <string>
#include <cstdio>

class Database {
private:
    char* connectionString;
    char* buffer;
    int connectionId;
    bool connected;

public:
    Database(const char* connStr) {
        connectionString = new char[256];
        strcpy(connectionString, connStr);
        buffer = (char*)malloc(1024);
        connectionId = 0;
        connected = false;
        std::cout << "Database object created\n";
    }

    ~Database() {
        std::cout << "Database object destroyed\n";
        // Memory leak: buffer is not freed
        // Memory leak: connectionString is not freed
    }

    bool connect() {
        char* tempBuffer = (char*)malloc(512);
        
        if (connectionString == nullptr) {
            std::cout << "Connection string is null\n";
            return false;  // Memory leak: tempBuffer not freed
        }

        strcpy(tempBuffer, "Connecting to: ");
        strcat(tempBuffer, connectionString);
        
        if (strlen(connectionString) < 5) {
            std::cout << "Invalid connection string\n";
            return false;  // Memory leak: tempBuffer not freed
        }

        connected = true;
        connectionId = 12345;
        free(tempBuffer);
        return true;
    }

    char* query(const char* sql) {
        char* resultSet = new char[2048];
        char* tempData = nullptr;
        
        if (!connected) {
            std::cout << "Not connected to database\n";
            return nullptr;  // Memory leak: resultSet not deleted
        }

        if (sql == nullptr) {
            std::cout << "SQL query is null\n";
            return nullptr;  // Memory leak: resultSet not deleted
        }

        strcpy(resultSet, "Results: ");
        strcpy(resultSet + 9, sql);  // Unsafe strcpy without bounds checking

        if (strlen(sql) > 100) {
            std::cout << "Query too long\n";
            return nullptr;  // Memory leak: resultSet not deleted
        }

        return resultSet;  // Only freed by caller, if at all
    }

    void executeStoredProc(const char* procName) {
        char* procBuffer = new char[512];
        char* paramBuffer = (char*)malloc(256);
        
        strcpy(procBuffer, procName);  // No bounds checking
        
        if (procName == nullptr) {
            return;  // Memory leak: both buffers not freed
        }

        if (!connected) {
            delete[] procBuffer;
            return;  // Memory leak: paramBuffer not freed
        }

        strcpy(paramBuffer, "exec ");
        strcat(paramBuffer, procName);

        std::cout << "Executing: " << paramBuffer << "\n";
        
        delete[] procBuffer;
        free(paramBuffer);
    }

    void setConnectionString(const char* newConnStr) {
        char* oldString = connectionString;
        connectionString = new char[256];
        strcpy(connectionString, newConnStr);  // No bounds checking
        // Memory leak: oldString not freed
    }

    int getStatus() {
        char* statusBuffer = nullptr;
        int status = statusBuffer[0];  // Null pointer dereference
        return status;
    }

    void processData(const char* data) {
        char* localBuffer;  // Uninitialized pointer
        
        if (data != nullptr) {
            strcpy(localBuffer, data);  // Using uninitialized pointer
        }
    }
};

int main() {
    Database* db = new Database("server=localhost;db=test");
    
    db->connect();
    
    char* results = db->query("SELECT * FROM users");
    if (results) {
        std::cout << results << "\n";
        // Memory leak: results not deleted
    }
    
    db->executeStoredProc("GetUserData");
    db->setConnectionString("server=remote;db=prod");
    
    int status = db->getStatus();
    
    delete db;
    
    return 0;
}