FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install TLA+ Toolbox and dependencies
RUN apt-get update && \
    apt-get install -y wget unzip default-jre && \
    wget https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar -O /usr/local/lib/tla2tools.jar && \
    echo '#!/bin/bash\njava -cp /usr/local/lib/tla2tools.jar tlc2.TLC "$@"' > /usr/local/bin/tlc && \
    chmod +x /usr/local/bin/tlc && \
    rm -rf /var/lib/apt/lists/*

# Create workspace directory
RUN mkdir -p /workspace

# Copy specification and configuration files
COPY two_phase_commit.tla /workspace/two_phase_commit.tla
COPY two_phase_commit.cfg /workspace/two_phase_commit.cfg

WORKDIR /workspace