#ifndef USER_H
#define USER_H

#include "../core/database.h"
#include "account.h"

#include <string>

class User {
private:
    int id;
    std::string name;
    std::string email;
    Account* account;
    Database* db;

public:
    User();
    User(int userId, const std::string& userName);
    ~User();

    int getId() const;
    std::string getName() const;
    void setName(const std::string& newName);
    
    std::string getEmail() const;
    void setEmail(const std::string& newEmail);
    
    Account* getAccount() const;
    void setAccount(Account* acc);
    
    bool save();
    bool load(int userId);
    bool remove();
};

#endif