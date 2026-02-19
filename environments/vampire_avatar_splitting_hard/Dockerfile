FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Create necessary directories
RUN mkdir -p /workspace/logs /workspace/solution

# Copy vampire help documentation
COPY data/vampire_help.txt /workspace/vampire_help.txt

# Copy all log files
COPY data/logs/run_001.log /workspace/logs/run_001.log
COPY data/logs/run_002.log /workspace/logs/run_002.log
COPY data/logs/run_003.log /workspace/logs/run_003.log
COPY data/logs/run_004.log /workspace/logs/run_004.log
COPY data/logs/run_005.log /workspace/logs/run_005.log
COPY data/logs/run_006.log /workspace/logs/run_006.log
COPY data/logs/run_007.log /workspace/logs/run_007.log
COPY data/logs/run_008.log /workspace/logs/run_008.log
COPY data/logs/run_009.log /workspace/logs/run_009.log
COPY data/logs/run_010.log /workspace/logs/run_010.log
COPY data/logs/run_011.log /workspace/logs/run_011.log
COPY data/logs/run_012.log /workspace/logs/run_012.log
COPY data/logs/run_013.log /workspace/logs/run_013.log
COPY data/logs/run_014.log /workspace/logs/run_014.log
COPY data/logs/run_015.log /workspace/logs/run_015.log

WORKDIR /workspace