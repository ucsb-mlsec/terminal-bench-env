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
    apt-get install -y --no-install-recommends \
    vim \
    less && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages pyyaml

RUN pip3 install --no-cache-dir --break-system-packages \
    pyyaml \
    jsonschema

RUN mkdir -p /workspace/config /workspace/data

COPY config/spark-application-current.yaml /workspace/config/spark-application-current.yaml
COPY data/executor-metrics-day1.json /workspace/data/executor-metrics-day1.json
COPY data/executor-metrics-day2.json /workspace/data/executor-metrics-day2.json
COPY data/executor-metrics-day3.json /workspace/data/executor-metrics-day3.json
COPY data/pod-events.json /workspace/data/pod-events.json
COPY data/cluster-resources.json /workspace/data/cluster-resources.json

WORKDIR /workspace