#ifndef DATABASE_H
#define DATABASE_H

#include <string>
#include <vector>
#include <memory>

namespace db {

class Connection;
class ResultSet;

class Database {
public:
    Database();
    explicit Database(const std::string& connectionString);
    ~Database();

    bool connect(const std::string& host, int port, const std::string& user,
                 const std::string& password);
    void disconnect();
    bool isConnected() const;

    std::shared_ptr<ResultSet> executeQuery(const std::string& query);
    bool executeUpdate(const std::string& query);
    bool executeBatch(const std::vector<std::string>& queries);

    void beginTransaction();
    void commit();
    void rollback();

    std::string getLastError() const;
    int getAffectedRows() const;

    void setTimeout(int seconds);
    void setAutoCommit(bool enable);

private:
    bool validateConnection();
    void logError(const std::string& error);

    std::shared_ptr<Connection> connection_;
    std::string connectionString_;
    std::string lastError_;
    int affectedRows_;
    int timeout_;
    bool autoCommit_;
    bool connected_;
};

}  // namespace db

#endif  // DATABASE_H