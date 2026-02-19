#ifndef CLIENT_HPP
#define CLIENT_HPP

#include "server.hpp"
#include "../core/logger.h"

namespace Network {

class Client {
public:
    Client();
    ~Client();
    
    bool connect(const char* address, int port);
    void disconnect();
    
    int send(const char* data, size_t length);
    int receive(char* buffer, size_t maxLength);
    
    bool isConnected() const;
    const char* getLastError() const;
    
private:
    void* m_socket;
    bool m_connected;
    char m_errorBuffer[256];
    Logger* m_logger;
};

} // namespace Network

#endif // CLIENT_HPP