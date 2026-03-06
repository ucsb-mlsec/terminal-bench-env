#include <jni.h>
#include "com_sensor_SensorProcessor.h"

JNIEXPORT jint JNICALL Java_com_sensor_SensorProcessor_calculateChecksum
  (JNIEnv *env, jobject obj, jint sensorValue) {
    return sensorValue * 2;
}