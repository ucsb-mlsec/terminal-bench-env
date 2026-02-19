#ifndef NETWORK_H
#define NETWORK_H

#include <cstdint>
#include <string>

// Network constants
const int BUFFER_SIZE = 4096;
const int MAX_PACKET_SIZE = 65535;
const int DEFAULT_TIMEOUT = 5000;
const int MAX_CONNECTIONS = 100;

// Packet header structure
struct PacketHeader {
    uint32_t size;
    uint16_t type;
    uint64_t timestamp;
    uint32_t sequence;
};

// Function declarations
int initSocket(int port);
int sendPacket(int socket, const char* data, int length);
int receiveData(int socket, char* buffer, int maxSize);
void closeSocket(int socket);
bool isSocketValid(int socket);
int setSocketTimeout(int socket, int timeoutMs);
std::string getLastError();

#endif