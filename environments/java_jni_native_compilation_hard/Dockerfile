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
    openjdk-21-jdk \
    maven \
    gcc \
    make \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc g++ make openjdk-11-jdk file && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/sensor-processor/src/main/java/com/sensor \
    /workspace/sensor-processor/src/native \
    /workspace/sensor-processor/lib

COPY sensor-processor/src/main/java/com/sensor/SensorProcessor.java /workspace/sensor-processor/src/main/java/com/sensor/SensorProcessor.java
COPY sensor-processor/src/native/sensor.c /workspace/sensor-processor/src/native/sensor.c
COPY sensor-processor/Makefile /workspace/sensor-processor/Makefile
COPY sensor-processor/pom.xml /workspace/sensor-processor/pom.xml

WORKDIR /workspace/sensor-processor