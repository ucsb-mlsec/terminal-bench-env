#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <cstdint>

struct Vertex {
    float x;
    float y;
    float z;
};

int count_x = 0;
int count_y = 0;
int count_z = 0;

void processVertices(const Vertex* vertices, size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
        if (vertices[i].x > 0) {
            count_x++;
        }
        if (vertices[i].y > 0) {
            count_y++;
        }
        if (vertices[i].z > 0) {
            count_z++;
        }
    }
}

int main() {
    const size_t NUM_VERTICES = 50000;
    const int NUM_THREADS = 8;
    
    std::vector<Vertex> vertices(NUM_VERTICES);
    
    std::ifstream input("/workspace/data/vertices.dat", std::ios::binary);
    if (!input) {
        std::cerr << "Error opening file" << std::endl;
        return 1;
    }
    
    input.read(reinterpret_cast<char*>(vertices.data()), NUM_VERTICES * sizeof(Vertex));
    if (!input) {
        std::cerr << "Error reading file" << std::endl;
        return 1;
    }
    input.close();
    
    std::vector<std::thread> threads;
    size_t chunk_size = NUM_VERTICES / NUM_THREADS;
    
    for (int i = 0; i < NUM_THREADS; i++) {
        size_t start = i * chunk_size;
        size_t end = (i == NUM_THREADS - 1) ? NUM_VERTICES : (i + 1) * chunk_size;
        threads.emplace_back(processVertices, vertices.data(), start, end);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    std::cout << count_x << std::endl;
    std::cout << count_y << std::endl;
    std::cout << count_z << std::endl;
    
    return 0;
}