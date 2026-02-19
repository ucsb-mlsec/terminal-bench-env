FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/android_app/app/src/main/java/com/example/app

COPY android_app/build.gradle /workspace/android_app/build.gradle
COPY android_app/app/build.gradle /workspace/android_app/app/build.gradle
COPY android_app/app/src/main/AndroidManifest.xml /workspace/android_app/app/src/main/AndroidManifest.xml
COPY android_app/app/src/main/java/com/example/app/MyApplication.kt /workspace/android_app/app/src/main/java/com/example/app/MyApplication.kt
COPY android_app/app/proguard-rules.pro /workspace/android_app/app/proguard-rules.pro
COPY android_app/gradle.properties /workspace/android_app/gradle.properties
COPY android_app/settings.gradle /workspace/android_app/settings.gradle
COPY android_app/app/src/main/java/com/example/app/MainActivity.kt /workspace/android_app/app/src/main/java/com/example/app/MainActivity.kt

WORKDIR /workspace