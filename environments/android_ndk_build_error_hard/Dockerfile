FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Android NDK dependencies
RUN apt-get update && \
    apt-get install -y wget unzip build-essential file && \
    rm -rf /var/lib/apt/lists/*

# Download and install Android NDK r21e
RUN mkdir -p /opt && \
    cd /opt && \
    wget -q https://dl.google.com/android/repository/android-ndk-r21e-linux-x86_64.zip && \
    unzip -q android-ndk-r21e-linux-x86_64.zip && \
    mv android-ndk-r21e android-ndk && \
    rm android-ndk-r21e-linux-x86_64.zip

# Set NDK environment variable
ENV ANDROID_NDK_HOME=/opt/android-ndk
ENV PATH=$PATH:/opt/android-ndk

# Create project directory structure
RUN mkdir -p /workspace/android-jni-project/jni && \
    mkdir -p /workspace/android-jni-project/libs/arm64-v8a

# Copy project files
COPY workspace/android-jni-project/jni/native-lib.cpp /workspace/android-jni-project/jni/native-lib.cpp
COPY workspace/android-jni-project/jni/image_utils.cpp /workspace/android-jni-project/jni/image_utils.cpp
COPY workspace/android-jni-project/jni/image_utils.h /workspace/android-jni-project/jni/image_utils.h
COPY workspace/android-jni-project/jni/Android.mk /workspace/android-jni-project/jni/Android.mk
COPY workspace/android-jni-project/jni/Application.mk /workspace/android-jni-project/jni/Application.mk

WORKDIR /workspace