FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install NuSMV model checker
RUN apt-get update && \
    apt-get install -y wget build-essential && \
    wget https://nusmv.fbk.eu/distrib/NuSMV-2.6.0-linux64.tar.gz && \
    tar -xzf NuSMV-2.6.0-linux64.tar.gz && \
    mkdir -p /usr/local/bin && \
    cp NuSMV-2.6.0-Linux/bin/NuSMV /usr/local/bin/NuSMV && \
    chmod +x /usr/local/bin/NuSMV && \
    rm -rf NuSMV-2.6.0-linux64.tar.gz NuSMV-2.6.0-Linux && \
    rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /home/agent /tmp

# Copy the traffic controller model to the expected location
COPY traffic_controller.smv /home/agent/traffic_controller.smv

WORKDIR /home/agent