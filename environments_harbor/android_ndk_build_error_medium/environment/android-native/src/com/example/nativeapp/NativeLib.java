package com.example.nativeapp;

public class NativeLib {
    
    static {
        System.loadLibrary("nativelib");
    }
    
    public native String getStringFromNative();
    
    public native int addNumbers(int a, int b);
    
    public native double calculateSquareRoot(double value);
    
    public static void main(String[] args) {
        NativeLib lib = new NativeLib();
        System.out.println(lib.getStringFromNative());
        System.out.println("5 + 3 = " + lib.addNumbers(5, 3));
    }
}