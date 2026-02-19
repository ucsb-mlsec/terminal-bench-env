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
    openjdk-17-jre-headless \
    wget \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y openjdk-11-jre-headless && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/cromwell && \
    wget -q -O /opt/cromwell/cromwell.jar https://github.com/broadinstitute/cromwell/releases/download/85/cromwell-85.jar

RUN mkdir -p /workspace/data/sequences && \
    mkdir -p /workspace/pipeline && \
    mkdir -p /workspace/results

COPY data/sequences/sample1.fasta /workspace/data/sequences/sample1.fasta
COPY data/sequences/sample2.fasta /workspace/data/sequences/sample2.fasta
COPY data/sequences/sample3.fasta /workspace/data/sequences/sample3.fasta
COPY pipeline/workflow.wdl /workspace/pipeline/workflow.wdl
COPY pipeline/inputs.json /workspace/pipeline/inputs.json
COPY pipeline/options.json /workspace/pipeline/options.json

WORKDIR /workspace