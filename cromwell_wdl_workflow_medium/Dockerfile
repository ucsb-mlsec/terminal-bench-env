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
    default-jdk \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y default-jre-headless && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/broadinstitute/cromwell/releases/download/85/cromwell-85.jar -O /usr/local/bin/cromwell.jar && \
    chmod +x /usr/local/bin/cromwell.jar

RUN mkdir -p /workspace/pipeline/data

COPY pipeline/process.wdl /workspace/pipeline/process.wdl
COPY pipeline/data/app.log /workspace/pipeline/data/app.log
COPY pipeline/data/system.log /workspace/pipeline/data/system.log

WORKDIR /workspace