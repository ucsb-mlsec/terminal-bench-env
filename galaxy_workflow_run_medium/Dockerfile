FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/data

COPY data/workflow1.json /workspace/data/workflow1.json
COPY data/workflow2.json /workspace/data/workflow2.json
COPY data/workflow3.json /workspace/data/workflow3.json
COPY data/invalid_workflow.json /workspace/data/invalid_workflow.json
COPY data/incomplete.json /workspace/data/incomplete.json
COPY data/empty_name.json /workspace/data/empty_name.json
COPY data/proteomics_workflow.json /workspace/data/proteomics_workflow.json
COPY data/wrong_version.json /workspace/data/wrong_version.json
COPY data/malformed.json /workspace/data/malformed.json

WORKDIR /workspace