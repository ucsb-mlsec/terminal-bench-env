#include "image_utils.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace ImageUtils {

void convertToGrayscale(unsigned char* data, int width, int height) {
    if (data == nullptr || width <= 0 || height <= 0) {
        return;
    }
    
    int totalPixels = width * height;
    for (int i = 0; i < totalPixels; i++) {
        int offset = i * 4; // Assuming RGBA format
        unsigned char r = data[offset];
        unsigned char g = data[offset + 1];
        unsigned char b = data[offset + 2];
        
        // Using luminosity method for grayscale conversion
        unsigned char gray = static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b);
        
        data[offset] = gray;
        data[offset + 1] = gray;
        data[offset + 2] = gray;
        // Alpha channel remains unchanged
    }
}

void adjustBrightness(unsigned char* data, int width, int height, int adjustment) {
    if (data == nullptr || width <= 0 || height <= 0) {
        return;
    }
    
    int totalPixels = width * height;
    for (int i = 0; i < totalPixels; i++) {
        int offset = i * 4; // Assuming RGBA format
        
        for (int c = 0; c < 3; c++) { // Only adjust RGB, not alpha
            int value = static_cast<int>(data[offset + c]) + adjustment;
            if (value > 255) value = 255;
            if (value < 0) value = 0;
            data[offset + c] = static_cast<unsigned char>(value);
        }
    }
}

void applyGaussianBlur(unsigned char* data, int width, int height, int radius) {
    if (data == nullptr || width <= 0 || height <= 0 || radius <= 0) {
        return;
    }
    
    // Create a simple Gaussian kernel
    int kernelSize = radius * 2 + 1;
    std::vector<float> kernel(kernelSize);
    float sigma = radius / 2.0f;
    float sum = 0.0f;
    
    for (int i = 0; i < kernelSize; i++) {
        int x = i - radius;
        kernel[i] = std::exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }
    
    // Normalize kernel
    for (int i = 0; i < kernelSize; i++) {
        kernel[i] /= sum;
    }
    
    // Create temporary buffer
    std::vector<unsigned char> temp(width * height * 4);
    std::memcpy(temp.data(), data, width * height * 4);
    
    // Horizontal pass
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float r = 0.0f, g = 0.0f, b = 0.0f;
            
            for (int k = 0; k < kernelSize; k++) {
                int sampleX = x + k - radius;
                if (sampleX < 0) sampleX = 0;
                if (sampleX >= width) sampleX = width - 1;
                
                int offset = (y * width + sampleX) * 4;
                r += temp[offset] * kernel[k];
                g += temp[offset + 1] * kernel[k];
                b += temp[offset + 2] * kernel[k];
            }
            
            int offset = (y * width + x) * 4;
            data[offset] = static_cast<unsigned char>(r);
            data[offset + 1] = static_cast<unsigned char>(g);
            data[offset + 2] = static_cast<unsigned char>(b);
        }
    }
    
    std::memcpy(temp.data(), data, width * height * 4);
    
    // Vertical pass
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float r = 0.0f, g = 0.0f, b = 0.0f;
            
            for (int k = 0; k < kernelSize; k++) {
                int sampleY = y + k - radius;
                if (sampleY < 0) sampleY = 0;
                if (sampleY >= height) sampleY = height - 1;
                
                int offset = (sampleY * width + x) * 4;
                r += temp[offset] * kernel[k];
                g += temp[offset + 1] * kernel[k];
                b += temp[offset + 2] * kernel[k];
            }
            
            int offset = (y * width + x) * 4;
            data[offset] = static_cast<unsigned char>(r);
            data[offset + 1] = static_cast<unsigned char>(g);
            data[offset + 2] = static_cast<unsigned char>(b);
        }
    }
}

void rotateImage90(unsigned char* input, unsigned char* output, int width, int height) {
    if (input == nullptr || output == nullptr || width <= 0 || height <= 0) {
        return;
    }
    
    // Rotate 90 degrees clockwise
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int srcOffset = (y * width + x) * 4;
            int dstX = height - 1 - y;
            int dstY = x;
            int dstOffset = (dstY * height + dstX) * 4;
            
            output[dstOffset] = input[srcOffset];
            output[dstOffset + 1] = input[srcOffset + 1];
            output[dstOffset + 2] = input[srcOffset + 2];
            output[dstOffset + 3] = input[srcOffset + 3];
        }
    }
}

} // namespace ImageUtils