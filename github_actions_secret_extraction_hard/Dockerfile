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
    file \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/audit/data /workspace/audit/data/.workflows

COPY data/workflow1_build.log /workspace/audit/data/workflow1_build.log
COPY data/workflow2_deploy.log /workspace/audit/data/workflow2_deploy.log
COPY data/workflow3_integration.log /workspace/audit/data/workflow3_integration.log
COPY data/workflow4_setup.log /workspace/audit/data/workflow4_setup.log
COPY data/workflow5_backup.log /workspace/audit/data/workflow5_backup.log
COPY data/workflow6_test.log /workspace/audit/data/workflow6_test.log
COPY data/workflow7_release.log /workspace/audit/data/workflow7_release.log
COPY data/workflow8_monitor.log /workspace/audit/data/workflow8_monitor.log
COPY data/metadata.json /workspace/audit/data/metadata.json
COPY data/README.txt /workspace/audit/data/README.txt
COPY data/.workflows/build.yml /workspace/audit/data/.workflows/build.yml
COPY data/.workflows/deploy.yml /workspace/audit/data/.workflows/deploy.yml
COPY data/.workflows/integration.yml /workspace/audit/data/.workflows/integration.yml
COPY data/.workflows/setup.yml /workspace/audit/data/.workflows/setup.yml

WORKDIR /workspace