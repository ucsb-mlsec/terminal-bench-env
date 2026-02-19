LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := native-lib

LOCAL_SRCS := native-lib.c

LOCAL_CFLAGS := -Wall -Wextra

LOCAL_LDLIBS := -llog

include $(BUILD_SHARED_LIBRARY)