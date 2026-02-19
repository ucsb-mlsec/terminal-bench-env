FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /tmp/helm_state /tmp

COPY data/release_metadata.json /tmp/helm_state/release_metadata.json
COPY data/deployed_resources.json /tmp/helm_state/deployed_resources.json
COPY data/helm_resources.json /tmp/helm_state/helm_resources.json
COPY data/pod_status.json /tmp/helm_state/pod_status.json

WORKDIR /tmp