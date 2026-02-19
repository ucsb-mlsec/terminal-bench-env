#ifndef HANDLER_H
#define HANDLER_H

#include "../models/user.h"
#include "../network/client.hpp"

#include <string>
#include <memory>

class Handler {
public:
    Handler();
    ~Handler();
    
    bool handleRequest(const std::string& request);
    void sendResponse(const std::string& response);
    
    void setUser(const User& user);
    User getCurrentUser() const;
    
    bool authenticate(const std::string& token);
    void disconnect();
    
    std::shared_ptr<Client> getClient() const;
    void setClient(std::shared_ptr<Client> client);

private:
    User currentUser;
    std::shared_ptr<Client> networkClient;
    bool isAuthenticated;
};

#endif