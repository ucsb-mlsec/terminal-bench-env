FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/decompiled_app

COPY decompiled_app/MainActivity.java /workspace/decompiled_app/MainActivity.java
COPY decompiled_app/DataCollectionService.java /workspace/decompiled_app/DataCollectionService.java
COPY decompiled_app/NetworkManager.java /workspace/decompiled_app/NetworkManager.java
COPY decompiled_app/ObfuscationHelper.java /workspace/decompiled_app/ObfuscationHelper.java
COPY decompiled_app/PermissionHelper.java /workspace/decompiled_app/PermissionHelper.java
COPY decompiled_app/AndroidManifest.xml /workspace/decompiled_app/AndroidManifest.xml
COPY decompiled_app/DeviceInfoCollector.java /workspace/decompiled_app/DeviceInfoCollector.java
COPY decompiled_app/SmsReader.java /workspace/decompiled_app/SmsReader.java
COPY decompiled_app/LocationTracker.java /workspace/decompiled_app/LocationTracker.java
COPY decompiled_app/ConfigManager.java /workspace/decompiled_app/ConfigManager.java

WORKDIR /workspace