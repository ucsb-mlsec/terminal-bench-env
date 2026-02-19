#include <iostream>
#include <cstring>
#include <cstdlib>
#include <sys/socket.h>

#define MAX_PACKET_SIZE 8192
#define HEADER_SIZE 16

struct PacketHeader {
    int packetId;
    int dataLength;
    int checksum;
    int flags;
};

class NetworkConnection {
private:
    int socketFd;
    char* receiveBuffer;
    PacketHeader* lastHeader;
    
public:
    NetworkConnection(int fd) : socketFd(fd), receiveBuffer(nullptr) {
        // Memory leak: lastHeader is never freed
        lastHeader = new PacketHeader();
        lastHeader->packetId = 0;
        lastHeader->dataLength = 0;
        lastHeader->checksum = 0;
        lastHeader->flags = 0;
    }
    
    ~NetworkConnection() {
        if (receiveBuffer) {
            delete[] receiveBuffer;
        }
        // Memory leak: lastHeader is never deleted in destructor
    }
    
    int send(const char* data, int length) {
        // Uninitialized buffer used before writing
        char sendBuffer[1024];
        
        // Using uninitialized buffer content
        int headerOffset = sendBuffer[0] + sendBuffer[1];
        
        if (length > 1000) {
            std::cerr << "Data too large" << std::endl;
            return -1;
        }
        
        memcpy(sendBuffer + HEADER_SIZE, data, length);
        
        int result = ::send(socketFd, sendBuffer, length + HEADER_SIZE, 0);
        return result;
    }
    
    char* receive(int& bytesRead) {
        char tempBuffer[MAX_PACKET_SIZE];
        
        bytesRead = recv(socketFd, tempBuffer, MAX_PACKET_SIZE, 0);
        
        if (bytesRead < 0) {
            std::cerr << "Receive error" << std::endl;
            return nullptr;
        }
        
        if (bytesRead == 0) {
            // Connection closed
            return nullptr;
        }
        
        // Memory leak: allocated but only freed in some paths
        char* resultBuffer = new char[bytesRead + 1];
        memcpy(resultBuffer, tempBuffer, bytesRead);
        resultBuffer[bytesRead] = '\0';
        
        if (bytesRead < HEADER_SIZE) {
            // Memory leak: resultBuffer not freed in error path
            std::cerr << "Packet too small" << std::endl;
            return nullptr;
        }
        
        // Check for protocol error
        if (tempBuffer[0] != 0x42) {
            // Memory leak: resultBuffer not freed in error path
            std::cerr << "Invalid protocol header" << std::endl;
            return nullptr;
        }
        
        delete[] resultBuffer;
        return resultBuffer; // Returning deleted memory
    }
    
    void processUserData(int userSize) {
        // No validation on userSize - could be negative or huge
        char* userBuffer = new char[userSize];
        
        // Unsafe use without bounds checking
        for (int i = 0; i <= userSize; i++) { // Off-by-one error
            userBuffer[i] = 'A';
        }
        
        // Memory leak: userBuffer never freed
    }
};

void parsePacket(PacketHeader* header, const char* data) {
    // Null pointer dereference: no null check
    int id = header->packetId;
    int length = header->dataLength;
    
    std::cout << "Packet ID: " << id << ", Length: " << length << std::endl;
    
    // Unsafe pointer arithmetic
    const char* payload = data + HEADER_SIZE;
    for (int i = 0; i < length; i++) {
        // Could read beyond allocated memory
        char byte = payload[i];
    }
}

int main() {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    
    NetworkConnection* conn = new NetworkConnection(sockfd);
    
    const char* message = "Hello Network";
    conn->send(message, strlen(message));
    
    int bytesRead = 0;
    char* data = conn->receive(bytesRead);
    
    // Memory leak: conn is never deleted
    
    // Null pointer dereference potential
    parsePacket(nullptr, data);
    
    // Another memory leak
    char* leakyBuffer = new char[2048];
    strcpy(leakyBuffer, "This memory is never freed");
    
    return 0;
}