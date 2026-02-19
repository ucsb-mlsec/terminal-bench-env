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
    apt-get install -y openjdk-11-jdk && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y default-jdk && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/agent /tmp

COPY VulnerableApp.java /home/agent/VulnerableApp.java
COPY data/test-input.ser /home/agent/test-input.ser
COPY commons-collections-3.2.1.jar /home/agent/commons-collections-3.2.1.jar

WORKDIR /home/agent

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
