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
    apt-get install -y systemd && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y iproute2 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/systemd/system /usr/local/bin /root

COPY etc/systemd/system/webapp.service /etc/systemd/system/webapp.service
COPY usr/local/bin/webapp /usr/local/bin/webapp

RUN chmod +x /usr/local/bin/webapp

WORKDIR /root