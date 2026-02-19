FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages pyyaml

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN pip3 install --no-cache-dir --break-system-packages pytest vcrpy requests

RUN mkdir -p /workspace/cassettes

COPY test_api.py /workspace/test_api.py
COPY cassettes/test_weather.yaml /workspace/cassettes/test_weather.yaml
COPY cassettes/test_users.yaml /workspace/cassettes/test_users.yaml
COPY cassettes/test_posts.yaml /workspace/cassettes/test_posts.yaml

WORKDIR /workspace