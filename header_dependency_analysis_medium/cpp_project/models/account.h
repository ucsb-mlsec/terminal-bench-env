#ifndef ACCOUNT_H
#define ACCOUNT_H

#include "user.h"
#include <string>

class Account {
private:
    std::string accountId;
    double balance;
    User* owner;
    bool isActive;

public:
    Account();
    Account(const std::string& id, User* user);
    ~Account();
    
    double getBalance() const;
    bool deposit(double amount);
    bool withdraw(double amount);
    
    void setOwner(User* user);
    User* getOwner() const;
    
    std::string getAccountId() const;
    void setAccountId(const std::string& id);
    
    bool isAccountActive() const;
    void activateAccount();
    void deactivateAccount();
};

#endif