FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN apt-get update && \
    apt-get install -y \
    file \
    binutils \
    openjdk-21-jdk \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y file binutils && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/native-libs \
    /usr/local/lib/jni \
    /tmp/legacy-libs \
    /opt/app/src \
    /opt/app/bin \
    /solution

COPY opt/native-libs/libauth_x86.so /opt/native-libs/libauth_x86.so
COPY opt/native-libs/libdata_arm64.so /opt/native-libs/libdata_arm64.so
COPY opt/native-libs/libnetwork.so.2.1 /opt/native-libs/libnetwork.so.2.1
COPY usr/local/lib/jni/libcrypto.so.1.1 /usr/local/lib/jni/libcrypto.so.1.1
COPY usr/local/lib/jni/libdata_proc_x64.so /usr/local/lib/jni/libdata_proc_x64.so
COPY usr/local/lib/jni/libssl_aarch64.so /usr/local/lib/jni/libssl_aarch64.so
COPY tmp/legacy-libs/libauth.so /tmp/legacy-libs/libauth.so
COPY tmp/legacy-libs/libjson_parser.so /tmp/legacy-libs/libjson_parser.so
COPY tmp/legacy-libs/libnetwork_arm.so /tmp/legacy-libs/libnetwork_arm.so
COPY tmp/legacy-libs/libutils_x86_64.so /tmp/legacy-libs/libutils_x86_64.so
COPY opt/app/src/AuthService.java /opt/app/src/AuthService.java
COPY opt/app/src/DataProcessor.java /opt/app/src/DataProcessor.java
COPY opt/app/src/NetworkHandler.java /opt/app/src/NetworkHandler.java
COPY opt/app/bin/start-services.sh /opt/app/bin/start-services.sh

RUN chmod +x /opt/app/bin/start-services.sh

WORKDIR /opt/app