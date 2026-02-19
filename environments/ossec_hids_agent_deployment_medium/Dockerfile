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
    python3 \
    python3-pip \
    libxml2-utils \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/ossec-agents /tmp

COPY config/agent-web01.conf /opt/ossec-agents/agent-web01.conf
COPY config/agent-web02.conf /opt/ossec-agents/agent-web02.conf
COPY config/agent-db01.conf /opt/ossec-agents/agent-db01.conf
COPY config/agent-app01.conf /opt/ossec-agents/agent-app01.conf
COPY config/agent-app02.conf /opt/ossec-agents/agent-app02.conf
COPY config/agent-cache01.conf /opt/ossec-agents/agent-cache01.conf

WORKDIR /opt/ossec-agents