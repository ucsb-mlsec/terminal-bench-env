// Native interface class for C++ library integration
// This class provides JNI bindings for native operations including
// image processing, encryption/decryption, and device information retrieval
package com.example.myapp.native_code;

public class NativeLib {
    
    static {
        System.loadLibrary("native-lib");
    }
    
    public native byte[] processImage(byte[] imageData, int width, int height);
    
    public native String encryptData(String input, String key);
    
    public native String decryptData(String encrypted, String key);
    
    public native boolean initializeEngine(String configPath);
    
    public native void cleanupResources();
    
    public native String getDeviceInfo();
}