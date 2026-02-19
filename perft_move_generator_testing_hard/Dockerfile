FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/validation_logs

COPY data/reference_results.txt /workspace/reference_results.txt
COPY data/validation_logs/test_run_001.log /workspace/validation_logs/test_run_001.log
COPY data/validation_logs/test_run_002.log /workspace/validation_logs/test_run_002.log
COPY data/validation_logs/test_run_003.log /workspace/validation_logs/test_run_003.log
COPY data/validation_logs/test_run_004.log /workspace/validation_logs/test_run_004.log
COPY data/validation_logs/test_run_005.log /workspace/validation_logs/test_run_005.log

WORKDIR /workspace