FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/rewrite_rules

COPY data/rewrite_rules/arithmetic.json /workspace/rewrite_rules/arithmetic.json
COPY data/rewrite_rules/basic.json /workspace/rewrite_rules/basic.json
COPY data/rewrite_rules/complex.json /workspace/rewrite_rules/complex.json
COPY data/rewrite_rules/identity.json /workspace/rewrite_rules/identity.json
COPY data/rewrite_rules/list_ops.json /workspace/rewrite_rules/list_ops.json
COPY data/rewrite_rules/simple.json /workspace/rewrite_rules/simple.json
COPY data/rewrite_rules/symmetric.json /workspace/rewrite_rules/symmetric.json

WORKDIR /workspace