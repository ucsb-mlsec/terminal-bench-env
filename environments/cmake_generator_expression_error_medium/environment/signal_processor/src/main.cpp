#include <iostream>
#include "signal_lib.h"

int main() {
    std::cout << "=== Signal Processor Tool ===" << std::endl;
    std::cout << "Library Version: " << getLibraryVersion() << std::endl;
    std::cout << std::endl;

    // Create sample audio data
    const int sampleCount = 16;
    float samples[sampleCount] = {
        0.0f, 0.5f, 0.8f, 1.0f,
        0.8f, 0.5f, 0.0f, -0.5f,
        -0.8f, -1.0f, -0.8f, -0.5f,
        0.0f, 0.3f, 0.6f, 0.4f
    };

    std::cout << "Original signal values:" << std::endl;
    for (int i = 0; i < sampleCount; i++) {
        std::cout << "  Sample[" << i << "]: " << samples[i] << std::endl;
    }

    // Process the signal using the library
    std::cout << std::endl << "Processing signal..." << std::endl;
    processSignal(samples, sampleCount);

    std::cout << std::endl << "Processed signal values:" << std::endl;
    for (int i = 0; i < sampleCount; i++) {
        std::cout << "  Sample[" << i << "]: " << samples[i] << std::endl;
    }

    std::cout << std::endl << "Signal processing completed successfully!" << std::endl;

    return 0;
}