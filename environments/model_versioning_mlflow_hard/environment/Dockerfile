FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /tmp/legacy_models /tmp

COPY data/legacy_models/model_002.json /tmp/legacy_models/model_002.json
COPY data/legacy_models/model_005.json /tmp/legacy_models/model_005.json
COPY data/legacy_models/model_006.json /tmp/legacy_models/model_006.json
COPY data/legacy_models/model_011.json /tmp/legacy_models/model_011.json
COPY data/legacy_models/model_012.json /tmp/legacy_models/model_012.json
COPY data/legacy_models/model_014.json /tmp/legacy_models/model_014.json
COPY data/legacy_models/model_015.json /tmp/legacy_models/model_015.json
COPY data/legacy_models/model_016.json /tmp/legacy_models/model_016.json
COPY data/legacy_models/model_018.json /tmp/legacy_models/model_018.json

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
