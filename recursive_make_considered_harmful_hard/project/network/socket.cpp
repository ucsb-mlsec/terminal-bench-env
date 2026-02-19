#include "socket.h"
#include "../common/utils.h"
#include <iostream>
#include <string>

namespace network {

Socket::Socket(const std::string& host, int port) 
    : hostname(host), port(port), connected(false) {
    std::string msg = "Socket created for " + host + ":" + std::to_string(port);
    utils::log(msg);
}

Socket::~Socket() {
    if (connected) {
        disconnect();
    }
    utils::log("Socket destroyed");
}

bool Socket::connect() {
    if (connected) {
        utils::log("Already connected");
        return true;
    }
    
    std::string address = hostname + ":" + std::to_string(port);
    if (!utils::validateString(address)) {
        utils::log("Invalid address format");
        return false;
    }
    
    connected = true;
    utils::log("Connected to " + address);
    return true;
}

void Socket::disconnect() {
    if (!connected) {
        return;
    }
    
    connected = false;
    utils::log("Disconnected from " + hostname);
}

bool Socket::send(const std::string& data) {
    if (!connected) {
        utils::log("Cannot send: not connected");
        return false;
    }
    
    if (!utils::validateString(data)) {
        utils::log("Invalid data to send");
        return false;
    }
    
    std::string formatted = utils::formatMessage(data);
    utils::log("Sent: " + formatted);
    return true;
}

std::string Socket::receive() {
    if (!connected) {
        utils::log("Cannot receive: not connected");
        return "";
    }
    
    std::string received = "mock_data_from_network";
    std::string formatted = utils::formatMessage(received);
    utils::log("Received: " + formatted);
    return formatted;
}

bool Socket::isConnected() const {
    return connected;
}

} // namespace network