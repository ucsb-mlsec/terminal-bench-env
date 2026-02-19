FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/experiment_data/run_001 \
    /workspace/experiment_data/run_002 \
    /workspace/experiment_data/run_003

RUN pip3 install --no-cache-dir --break-system-packages mlflow

COPY data/experiment_data/run_001/metrics.csv /workspace/experiment_data/run_001/metrics.csv
COPY data/experiment_data/run_001/predictions.csv /workspace/experiment_data/run_001/predictions.csv
COPY data/experiment_data/run_001/config.json /workspace/experiment_data/run_001/config.json
COPY data/experiment_data/run_002/metrics.csv /workspace/experiment_data/run_002/metrics.csv
COPY data/experiment_data/run_002/predictions.csv /workspace/experiment_data/run_002/predictions.csv
COPY data/experiment_data/run_002/config.json /workspace/experiment_data/run_002/config.json
COPY data/experiment_data/run_003/metrics.csv /workspace/experiment_data/run_003/metrics.csv
COPY data/experiment_data/run_003/predictions.csv /workspace/experiment_data/run_003/predictions.csv
COPY data/experiment_data/run_003/config.json /workspace/experiment_data/run_003/config.json

WORKDIR /workspace