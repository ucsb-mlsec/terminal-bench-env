#include "signal_lib.h"
#include <cmath>
#include <algorithm>

// Library version information
const char* getLibraryVersion() {
    return "1.0";
}

// Simple signal normalization function
void processSignal(float* buffer, int size) {
    if (buffer == nullptr || size <= 0) {
        return;
    }
    
    // Find maximum absolute value
    float maxVal = 0.0f;
    for (int i = 0; i < size; ++i) {
        float absVal = std::abs(buffer[i]);
        if (absVal > maxVal) {
            maxVal = absVal;
        }
    }
    
    // Normalize if needed
    if (maxVal > 0.0f) {
        float scale = 1.0f / maxVal;
        for (int i = 0; i < size; ++i) {
            buffer[i] *= scale;
        }
    }
}

// Apply a simple low-pass filter
void applyLowPassFilter(float* buffer, int size, float cutoff) {
    if (buffer == nullptr || size <= 1) {
        return;
    }
    
    // Simple moving average filter
    float alpha = std::max(0.0f, std::min(1.0f, cutoff));
    float prev = buffer[0];
    
    for (int i = 1; i < size; ++i) {
        buffer[i] = alpha * buffer[i] + (1.0f - alpha) * prev;
        prev = buffer[i];
    }
}