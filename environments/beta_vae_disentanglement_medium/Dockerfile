FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/experiments/run_001 \
    /workspace/experiments/run_002 \
    /workspace/experiments/run_003 \
    /workspace/experiments/run_004 \
    /workspace/experiments/run_005 \
    /workspace/experiments/run_006

COPY data/README.txt /workspace/README.txt
COPY data/experiments/run_001/config.json /workspace/experiments/run_001/config.json
COPY data/experiments/run_001/metrics.log /workspace/experiments/run_001/metrics.log
COPY data/experiments/run_001/final_scores.txt /workspace/experiments/run_001/final_scores.txt
COPY data/experiments/run_002/config.json /workspace/experiments/run_002/config.json
COPY data/experiments/run_002/metrics.log /workspace/experiments/run_002/metrics.log
COPY data/experiments/run_002/final_scores.txt /workspace/experiments/run_002/final_scores.txt
COPY data/experiments/run_003/config.json /workspace/experiments/run_003/config.json
COPY data/experiments/run_003/metrics.log /workspace/experiments/run_003/metrics.log
COPY data/experiments/run_003/final_scores.txt /workspace/experiments/run_003/final_scores.txt
COPY data/experiments/run_004/config.json /workspace/experiments/run_004/config.json
COPY data/experiments/run_004/metrics.log /workspace/experiments/run_004/metrics.log
COPY data/experiments/run_004/final_scores.txt /workspace/experiments/run_004/final_scores.txt
COPY data/experiments/run_005/config.json /workspace/experiments/run_005/config.json
COPY data/experiments/run_005/metrics.log /workspace/experiments/run_005/metrics.log
COPY data/experiments/run_005/final_scores.txt /workspace/experiments/run_005/final_scores.txt
COPY data/experiments/run_006/config.json /workspace/experiments/run_006/config.json
COPY data/experiments/run_006/metrics.log /workspace/experiments/run_006/metrics.log
COPY data/experiments/run_006/final_scores.txt /workspace/experiments/run_006/final_scores.txt

WORKDIR /workspace