FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/uge_archive/config /solution

COPY config/uge_archive/config/exec_hosts.conf /opt/uge_archive/config/exec_hosts.conf
COPY config/uge_archive/config/queue.conf /opt/uge_archive/config/queue.conf
COPY config/uge_archive/config/host_groups.conf /opt/uge_archive/config/host_groups.conf
COPY config/uge_archive/config/complex_config.conf /opt/uge_archive/config/complex_config.conf
COPY config/uge_archive/config/additional_hosts.conf /opt/uge_archive/config/additional_hosts.conf
COPY config/uge_archive/config/parallel_environments.conf /opt/uge_archive/config/parallel_environments.conf

WORKDIR /workspace