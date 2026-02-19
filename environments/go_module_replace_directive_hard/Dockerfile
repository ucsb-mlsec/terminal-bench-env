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
    apt-get install -y golang-go && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/app /workspace/logger /tmp

COPY logger/go.mod /workspace/logger/go.mod
COPY logger/logger.go /workspace/logger/logger.go
COPY app/go.mod /workspace/app/go.mod
COPY app/main.go /workspace/app/main.go

WORKDIR /workspace