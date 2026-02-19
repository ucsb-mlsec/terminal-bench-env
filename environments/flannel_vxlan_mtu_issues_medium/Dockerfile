FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/diagnostics /tmp

COPY data/host_interfaces.txt /var/diagnostics/host_interfaces.txt
COPY data/flannel_config.yaml /var/diagnostics/flannel_config.yaml
COPY data/pod_interfaces.txt /var/diagnostics/pod_interfaces.txt
COPY data/network_specs.txt /var/diagnostics/network_specs.txt

WORKDIR /workspace