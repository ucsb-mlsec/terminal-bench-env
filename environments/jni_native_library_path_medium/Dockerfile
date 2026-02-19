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
    openjdk-11-jdk \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y openjdk-11-jdk && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/jniapp /usr/local/lib/jniapp /solution

COPY opt/jniapp/NativeApp.java /opt/jniapp/NativeApp.java
COPY opt/jniapp/NativeApp.class /opt/jniapp/NativeApp.class
COPY opt/jniapp/run.sh /opt/jniapp/run.sh
COPY usr/local/lib/jniapp/libnativecalc.so /usr/local/lib/jniapp/libnativecalc.so

RUN chmod +x /opt/jniapp/run.sh

WORKDIR /opt/jniapp