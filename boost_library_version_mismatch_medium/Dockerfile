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
    apt-get install -y \
    build-essential \
    g++ \
    make \
    libboost1.74-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y g++ make libboost-all-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/json_config_reader

COPY main.cpp /workspace/json_config_reader/main.cpp
COPY Makefile /workspace/json_config_reader/Makefile
COPY data/sample.json /workspace/json_config_reader/sample.json

WORKDIR /workspace/json_config_reader