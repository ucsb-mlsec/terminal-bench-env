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
    systemctl \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/github-runner && \
    mkdir -p /tmp

RUN useradd -m -d /home/runner -s /bin/bash runner || true

COPY data/runner_binaries_placeholder.txt /opt/github-runner/

RUN chown -R runner:runner /opt/github-runner

WORKDIR /opt/github-runner