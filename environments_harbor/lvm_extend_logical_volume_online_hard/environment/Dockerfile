FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/lvm-sim /root

COPY config/pv_state.json /opt/lvm-sim/pv_state.json
COPY config/vg_state.json /opt/lvm-sim/vg_state.json
COPY config/lv_state.json /opt/lvm-sim/lv_state.json
COPY config/fs_state.json /opt/lvm-sim/fs_state.json

WORKDIR /root

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
