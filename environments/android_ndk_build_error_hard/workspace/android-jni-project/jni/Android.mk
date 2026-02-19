LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := native-lib

LOCAL_SRC_FILES := native-lib.cpp image_utils.cpp

LOCAL_CPPFLAGS := -std=c++98 -Wall

LOCAL_CFLAGS := -O2

include $(BUILD_SHARED_LIBRARY)