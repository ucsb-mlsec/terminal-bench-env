#include <jni.h>
#include <string>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

// Helper function to convert RGB to grayscale
void convertToGrayscale(unsigned char* pixels, int width, int height) {
    int totalPixels = width * height;
    for (int i = 0; i < totalPixels * 4; i += 4) {
        unsigned char r = pixels[i];
        unsigned char g = pixels[i + 1];
        unsigned char b = pixels[i + 2];
        unsigned char gray = static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b);
        pixels[i] = gray;
        pixels[i + 1] = gray;
        pixels[i + 2] = gray;
    }
}

// Helper function to adjust brightness
void adjustBrightness(unsigned char* pixels, int width, int height, int adjustment) {
    int totalPixels = width * height;
    for (int i = 0; i < totalPixels * 4; i++) {
        if (i % 4 == 3) continue; // Skip alpha channel
        int newValue = pixels[i] + adjustment;
        pixels[i] = static_cast<unsigned char>(std::max(0, std::min(255, newValue)));
    }
}

// Helper function for simple box blur
void applyBoxBlur(unsigned char* pixels, int width, int height, int radius) {
    std::vector<unsigned char> temp(width * height * 4);
    std::memcpy(temp.data(), pixels, width * height * 4);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int sumR = 0, sumG = 0, sumB = 0, count = 0;
            
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        int idx = (ny * width + nx) * 4;
                        sumR += temp[idx];
                        sumG += temp[idx + 1];
                        sumB += temp[idx + 2];
                        count++;
                    }
                }
            }
            
            int idx = (y * width + x) * 4;
            pixels[idx] = sumR / count;
            pixels[idx + 1] = sumG / count;
            pixels[idx + 2] = sumB / count;
        }
    }
}

extern "C" {

// JNI function to convert image to grayscale
JNIEXPORT void JNICALL
Java_com_example_app_NativeLib_convertToGrayscale(JNIEnv* env, jobject obj, 
                                                    jbyteArray imageData, 
                                                    jint width, jint height) {
    if (imageData == nullptr) {
        return;
    }
    
    jsize len = env->GetArrayLength(imageData);
    if (len != width * height * 4) {
        return;
    }
    
    jbyte* pixels = env->GetByteArrayElements(imageData, nullptr);
    if (pixels == nullptr) {
        return;
    }
    
    convertToGrayscale(reinterpret_cast<unsigned char*>(pixels), width, height);
    
    env->ReleaseByteArrayElements(imageData, pixels, 0);
}

// JNI function to adjust image brightness
JNIEXPORT void JNICALL
Java_com_example_app_NativeLib_adjustBrightness(JNIEnv* env, jobject obj,
                                                 jbyteArray imageData,
                                                 jint width, jint height,
                                                 jint adjustment) {
    if (imageData == nullptr) {
        return;
    }
    
    jsize len = env->GetArrayLength(imageData);
    if (len != width * height * 4) {
        return;
    }
    
    jbyte* pixels = env->GetByteArrayElements(imageData, nullptr);
    if (pixels == nullptr) {
        return;
    }
    
    adjustBrightness(reinterpret_cast<unsigned char*>(pixels), width, height, adjustment);
    
    env->ReleaseByteArrayElements(imageData, pixels, 0);
}

// JNI function to apply blur filter
JNIEXPORT void JNICALL
Java_com_example_app_NativeLib_applyBlur(JNIEnv* env, jobject obj,
                                          jbyteArray imageData,
                                          jint width, jint height,
                                          jint radius) {
    if (imageData == nullptr) {
        return;
    }
    
    jsize len = env->GetArrayLength(imageData);
    if (len != width * height * 4) {
        return;
    }
    
    if (radius < 1 || radius > 10) {
        return;
    }
    
    jbyte* pixels = env->GetByteArrayElements(imageData, nullptr);
    if (pixels == nullptr) {
        return;
    }
    
    applyBoxBlur(reinterpret_cast<unsigned char*>(pixels), width, height, radius);
    
    env->ReleaseByteArrayElements(imageData, pixels, 0);
}

// JNI function to get library version
JNIEXPORT jstring JNICALL
Java_com_example_app_NativeLib_getVersion(JNIEnv* env, jobject obj) {
    std::string version = "1.0.0";
    return env->NewStringUTF(version.c_str());
}

}