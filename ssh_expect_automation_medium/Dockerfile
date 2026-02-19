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
    apt-get install -y expect && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y tcl && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/mock_ssh/profiles /home/agent

COPY config/prod.conf /etc/mock_ssh/profiles/prod.conf
COPY config/staging.conf /etc/mock_ssh/profiles/staging.conf
COPY config/dev.conf /etc/mock_ssh/profiles/dev.conf
COPY mock_ssh_client /usr/local/bin/mock_ssh_client

RUN chmod +x /usr/local/bin/mock_ssh_client

WORKDIR /home/agent