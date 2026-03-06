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
    jq \
    bc && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/lib/jenkins/workspace/project1 \
    /var/lib/jenkins/workspace/project2 \
    /var/lib/jenkins/workspace/project3 \
    /var/lib/jenkins/workspace/project4 \
    /var/lib/jenkins/workspace/project5 \
    /var/lib/jenkins/workspace/project6 \
    /var/lib/jenkins/workspace/project7 \
    /var/lib/jenkins/workspace/project8 \
    /var/lib/jenkins/workspace/project9 \
    /var/lib/jenkins/workspace/project10 \
    /var/lib/jenkins/workspace/project11 \
    /var/lib/jenkins/workspace/project12 \
    /var/lib/jenkins/workspace/project13 \
    /data \
    /home/agent

COPY data/workspace1_metadata.json /var/lib/jenkins/workspace/project1/metadata.json
COPY data/workspace2_metadata.json /var/lib/jenkins/workspace/project2/metadata.json
COPY data/workspace3_metadata.json /var/lib/jenkins/workspace/project3/metadata.json
COPY data/workspace4_metadata.json /var/lib/jenkins/workspace/project4/metadata.json
COPY data/workspace5_metadata.json /var/lib/jenkins/workspace/project5/metadata.json
COPY data/workspace6_metadata.json /var/lib/jenkins/workspace/project6/metadata.json
COPY data/workspace7_metadata.json /var/lib/jenkins/workspace/project7/metadata.json
COPY data/workspace8_metadata.json /var/lib/jenkins/workspace/project8/metadata.json
COPY data/workspace9_metadata.json /var/lib/jenkins/workspace/project9/metadata.json
COPY data/workspace10_metadata.json /var/lib/jenkins/workspace/project10/metadata.json
COPY data/workspace11_metadata.json /var/lib/jenkins/workspace/project11/metadata.json
COPY data/workspace12_metadata.json /var/lib/jenkins/workspace/project12/metadata.json
COPY data/workspace13_metadata.json /var/lib/jenkins/workspace/project13/metadata.json
COPY data/disk_usage_before.txt /data/disk_usage_before.txt
COPY data/workspace_sizes.txt /data/workspace_sizes.txt

RUN for i in {1..13}; do \
    ws="/var/lib/jenkins/workspace/project$i"; \
    mkdir -p "$ws/build" "$ws/target" "$ws/node_modules" "$ws/.cache"; \
    dd if=/dev/zero of="$ws/build/artifact.jar" bs=1M count=50 2>/dev/null || true; \
    dd if=/dev/zero of="$ws/target/output.war" bs=1M count=40 2>/dev/null || true; \
    dd if=/dev/zero of="$ws/node_modules/package.tar" bs=1M count=60 2>/dev/null || true; \
    dd if=/dev/zero of="$ws/.cache/temp.tmp" bs=1M count=30 2>/dev/null || true; \
    dd if=/dev/zero of="$ws/source.java" bs=1M count=5 2>/dev/null || true; \
    done

WORKDIR /home/agent

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
