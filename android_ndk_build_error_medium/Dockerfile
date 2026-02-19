FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Android NDK and build dependencies
RUN apt-get update && \
    apt-get install -y \
    wget \
    unzip \
    build-essential \
    file && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y build-essential file && \
    rm -rf /var/lib/apt/lists/*

# Download and install Android NDK
RUN cd /opt && \
    wget -q https://dl.google.com/android/repository/android-ndk-r25c-linux.zip && \
    unzip -q android-ndk-r25c-linux.zip && \
    rm android-ndk-r25c-linux.zip && \
    mv android-ndk-r25c android-ndk

# Add NDK to PATH
ENV ANDROID_NDK_HOME=/opt/android-ndk
ENV PATH="${ANDROID_NDK_HOME}:${PATH}"

# Create necessary directories
RUN mkdir -p /workspace/android-native/jni && \
    mkdir -p /workspace/android-native/src/com/example/nativeapp

# Copy project files
COPY android-native/jni/native-lib.c /workspace/android-native/jni/native-lib.c
COPY android-native/jni/Android.mk /workspace/android-native/jni/Android.mk
COPY android-native/jni/Application.mk /workspace/android-native/jni/Application.mk
COPY android-native/src/com/example/nativeapp/NativeLib.java /workspace/android-native/src/com/example/nativeapp/NativeLib.java

WORKDIR /workspace/android-native