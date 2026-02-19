FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/agent/decompiled_app/com/example/app/util && \
    mkdir -p /home/agent/decompiled_app/com/example/app/network

COPY decompiled_app/com/example/app/CryptoUtils.java /home/agent/decompiled_app/com/example/app/CryptoUtils.java
COPY decompiled_app/com/example/app/NetworkHandler.java /home/agent/decompiled_app/com/example/app/NetworkHandler.java
COPY decompiled_app/com/example/app/DataStorage.java /home/agent/decompiled_app/com/example/app/DataStorage.java
COPY decompiled_app/com/example/app/AuthModule.java /home/agent/decompiled_app/com/example/app/AuthModule.java
COPY decompiled_app/com/example/app/util/EncryptionHelper.java /home/agent/decompiled_app/com/example/app/util/EncryptionHelper.java
COPY decompiled_app/com/example/app/SecurePreferences.java /home/agent/decompiled_app/com/example/app/SecurePreferences.java
COPY decompiled_app/com/example/app/network/ApiClient.java /home/agent/decompiled_app/com/example/app/network/ApiClient.java
COPY decompiled_app/com/example/app/MainActivity.java /home/agent/decompiled_app/com/example/app/MainActivity.java

WORKDIR /home/agent