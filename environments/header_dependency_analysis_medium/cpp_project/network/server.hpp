#ifndef SERVER_HPP
#define SERVER_HPP

#include "protocol.hpp"
#include "connection.h"

namespace network {

class Server {
public:
    Server();
    Server(int port);
    ~Server();
    
    bool start();
    void stop();
    bool isRunning() const;
    
    void acceptConnections();
    void handleClient(int clientSocket);
    
    void setPort(int port);
    int getPort() const;
    
    void setMaxConnections(int max);
    int getActiveConnections() const;
    
private:
    int serverSocket;
    int port;
    bool running;
    int maxConnections;
    int activeConnections;
};

} // namespace network

#endif // SERVER_HPP