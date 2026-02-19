#ifndef SOCKET_H
#define SOCKET_H

#include <string>
#include "../common/utils.h"

namespace network {

class Socket {
public:
    Socket();
    ~Socket();
    
    bool connect(const std::string& host, int port);
    void send(const std::string& data);
    std::string receive();
    void close();
    
private:
    int sockfd;
    bool connected;
};

} // namespace network

#endif // SOCKET_H