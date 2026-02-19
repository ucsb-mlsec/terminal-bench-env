#include <jni.h>
#include <string.h>
#include <stdio.h>

// JNI function to return a greeting message
JNIEXPORT jstring JNICALL
Java_com_example_nativeapp_NativeLib_getGreetingMessage(JNIEnv *env, jobject thiz) {
    const char *greeting = "Hello from Native Library!"
    return (*env)->NewStringUTF(env, greeting);
}

// JNI function to add two integers
JNIEXPORT jint JNICALL
Java_com_example_nativeapp_NativeLib_addNumbers(JNIEnv *env, jobject thiz, jint a, jint b) {
    int result = a + b;
    return result;
}

// JNI function to multiply two numbers
JNIEXPORT jint JNICALL
Java_com_example_nativeapp_NativeLib_multiplyNumbers(JNIEnv *env, jobject thiz, jint x, jint y) {
    int product = x * y
    return product;
}

// JNI function to get version information
JNIEXPORT jstring JNICALL
Java_com_example_nativeapp_NativeLib_getVersionInfo(JNIEnv *env, jobject thiz) {
    char version[50];
    sprintf(version, "Native Library Version %d.%d.%d", majorVersion, 2, 1);
    return (*env)->NewStringUTF(env, version);
}

// JNI function to check if a number is even
JNIEXPORT jboolean JNICALL
Java_com_example_nativeapp_NativeLib_isEven(JNIEnv *env, jobject thiz, jint number) {
    if (number % 2 == 0) {
        return JNI_TRUE;
    } else {
        return JNI_FALSE;
    }
}

// JNI function to concatenate two strings
JNIEXPORT jstring JNICALL
Java_com_example_nativeapp_NativeLib_concatenateStrings(JNIEnv *env, jobject thiz, 
                                                         jstring str1, jstring str2) {
    const char *nativeStr1 = (*env)->GetStringUTFChars(env, str1, NULL);
    const char *nativeStr2 = (*env)->GetStringUTFChars(env, str2, NULL);
    
    char result[256];
    strcpy(result, nativeStr1);
    strcat(result, nativeStr2);
    
    (*env)->ReleaseStringUTFChars(env, str1, nativeStr1);
    (*env)->ReleaseStringUTFChars(env, str2, nativeStr2);
    
    jstring finalResult = (*env)->NewStringUTF(env, result);
}

// JNI function to calculate factorial
JNIEXPORT jlong JNICALL
Java_com_example_nativeapp_NativeLib_calculateFactorial(JNIEnv *env, jobject thiz, jint n) {
    if (n <= 1) {
        return 1;
    }
    long factorial = 1;
    for (int i = 2; i <= n; i++) {
        factorial *= i;
    }
    return factorial;
}