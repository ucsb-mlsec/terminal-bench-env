#include <iostream>
#include <string>
#include "../common/utils.h"
#include "../network/socket.h"

int main() {
    std::cout << "Starting application..." << std::endl;
    
    // Use utils library functionality
    int result = utils::calculate(42, 10);
    std::cout << "Utils calculation result: " << result << std::endl;
    
    std::string formatted = utils::formatMessage("Hello from main");
    std::cout << "Formatted message: " << formatted << std::endl;
    
    // Use network library functionality
    network::Socket socket;
    std::cout << "Created socket object" << std::endl;
    
    bool connected = socket.connect("127.0.0.1", 8080);
    if (connected) {
        std::cout << "Socket connected successfully" << std::endl;
        socket.send("Test message");
        std::string response = socket.receive();
        std::cout << "Received: " << response << std::endl;
        socket.close();
    } else {
        std::cout << "Socket connection failed" << std::endl;
    }
    
    std::cout << "Application finished" << std::endl;
    return 0;
}