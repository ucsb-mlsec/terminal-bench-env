FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/decompiled_app/com/android/systemcore/network \
    /workspace/decompiled_app/com/android/systemcore/utils \
    /workspace/decompiled_app/com/android/systemcore/service \
    /workspace/decompiled_app/com/android/systemcore/receiver

COPY decompiled_app/com/android/systemcore/MainActivity.java /workspace/decompiled_app/com/android/systemcore/MainActivity.java
COPY decompiled_app/com/android/systemcore/network/NetworkManager.java /workspace/decompiled_app/com/android/systemcore/network/NetworkManager.java
COPY decompiled_app/com/android/systemcore/network/ApiClient.java /workspace/decompiled_app/com/android/systemcore/network/ApiClient.java
COPY decompiled_app/com/android/systemcore/utils/DeviceInfo.java /workspace/decompiled_app/com/android/systemcore/utils/DeviceInfo.java
COPY decompiled_app/com/android/systemcore/service/DataCollector.java /workspace/decompiled_app/com/android/systemcore/service/DataCollector.java
COPY decompiled_app/com/android/systemcore/Config.java /workspace/decompiled_app/com/android/systemcore/Config.java
COPY decompiled_app/com/android/systemcore/receiver/BootReceiver.java /workspace/decompiled_app/com/android/systemcore/receiver/BootReceiver.java
COPY decompiled_app/com/android/systemcore/utils/CryptoHelper.java /workspace/decompiled_app/com/android/systemcore/utils/CryptoHelper.java

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
