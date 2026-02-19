#include <jni.h>
#include <string.h>
#include <stdlib.h>

JNIEXPORT jstring JNICALL Java_StringManipulator_reverseString(JNIEnv *env, jobject obj, jstring input) {
    if (input == NULL) {
        return NULL;
    }
    
    const char *nativeString = (*env)->GetStringUTFChars(env, input, 0);
    if (nativeString == NULL) {
        return NULL;
    }
    
    int length = strlen(nativeString);
    char *reversed = (char *)malloc(length + 1);
    if (reversed == NULL) {
        (*env)->ReleaseStringUTFChars(env, input, nativeString);
        return NULL;
    }
    
    for (int i = 0; i < length; i++) {
        reversed[i] = nativeString[length - 1 - i];
    }
    reversed[length] = '\0';
    
    jstring result = (*env)->NewStringUTF(env, reversed);
    
    free(reversed);
    (*env)->ReleaseStringUTFChars(env, input, nativeString);
    
    return result;
}