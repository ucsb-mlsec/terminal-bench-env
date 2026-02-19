FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/slack_export/customer-support \
    /workspace/slack_export/general \
    /workspace/slack_export/engineering

COPY data/slack_export/users.json /workspace/slack_export/users.json
COPY data/slack_export/customer-support/2024-01-15.json /workspace/slack_export/customer-support/2024-01-15.json
COPY data/slack_export/customer-support/2024-01-16.json /workspace/slack_export/customer-support/2024-01-16.json
COPY data/slack_export/customer-support/2024-01-17.json /workspace/slack_export/customer-support/2024-01-17.json
COPY data/slack_export/customer-support/2024-01-19.json /workspace/slack_export/customer-support/2024-01-19.json
COPY data/slack_export/general/2024-01-15.json /workspace/slack_export/general/2024-01-15.json
COPY data/slack_export/engineering/2024-01-16.json /workspace/slack_export/engineering/2024-01-16.json

WORKDIR /workspace