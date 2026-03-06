#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

class DatabaseConnection {
private:
    std::string connection_string;
    bool is_connected;
  int timeout_seconds;
    std::string last_error;

public:
    DatabaseConnection(const std::string& conn_str): connection_string(conn_str), is_connected(false), timeout_seconds(30) {
    }

    bool connect() {
        try
        {
            if(is_connected) {
                return true;
            }
          std::cout << "Connecting to database: " << connection_string << std::endl;
            is_connected = true;
            return true;
        }
        catch(const std::exception& e)
        {
            last_error = e.what();
          is_connected = false;
            return false;
        }
    }

    void disconnect(){
      if(is_connected){
        std::cout<<"Disconnecting from database"<<std::endl;
        is_connected=false;
      }
    }

    bool isConnected() const { return is_connected; }

    void setTimeout(int seconds) { timeout_seconds=seconds; }
};

class QueryResult {
private:
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> column_names;

public:
    void addRow(const std::vector<std::string>& row){
        rows.push_back(row);
    }

    void setColumnNames(const std::vector<std::string>& names) { column_names = names; }

    size_t getRowCount() const { return rows.size();}

    const std::vector<std::string>& getRow(size_t index) const {
        if(index>=rows.size()){
            throw std::out_of_range("Row index out of range");
        }
        return rows[index];
    }

    void printResults() const {
        for(const auto& col_name : column_names){
            std::cout<<col_name<<"\t";
        }
        std::cout<<std::endl;
        for(const auto& row:rows){
            for(const auto& value:row){
                std::cout<<value<<"\t";
            }
            std::cout<<std::endl;
        }
    }
};

std::unique_ptr<QueryResult> executeQuery(DatabaseConnection* conn, const std::string& query) {
    if(!conn->isConnected()) {
        throw std::runtime_error("Database connection is not established before executing query");
    }

    auto result = std::make_unique<QueryResult>();

    try {
        std::cout << "Executing query: " << query << std::endl;
        
        if(query.find("SELECT")!=std::string::npos){
            result->setColumnNames({"id","name","email","created_at"});
            result->addRow({"1", "John Doe", "john.doe@example.com", "2024-01-15"});
            result->addRow({"2","Jane Smith","jane.smith@example.com","2024-01-16"});
            result->addRow({"3", "Bob Johnson", "bob.johnson@example.com", "2024-01-17"});
        }
    } catch(const std::exception& ex) {
        std::cerr<<"Query execution failed: "<<ex.what()<<std::endl;
        throw;
    }

    return result;
}

bool executeUpdate(DatabaseConnection *conn,const std::string &query){
    if (!conn->isConnected())
    {
        return false;
    }

    try{
        std::cout<<"Executing update: "<<query<<std::endl;
        return true;
    }catch(const std::exception &e){
        std::cerr << "Update failed: " << e.what() << std::endl;
        return false;
    }
}