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
    jq \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp /workspace/config /workspace/data /workspace/backend-api /workspace/data-processor /workspace/ml-service

COPY config/.gitlab-ci.yml /workspace/config/.gitlab-ci.yml
COPY data/pipeline_logs.json /workspace/data/pipeline_logs.json
COPY backend-api/requirements.txt /workspace/backend-api/requirements.txt
COPY data-processor/Cargo.toml /workspace/data-processor/Cargo.toml
COPY ml-service/build.gradle /workspace/ml-service/build.gradle
COPY data/cache_analysis.json /workspace/data/cache_analysis.json

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
