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
    apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/infra

COPY infra/index.ts /workspace/infra/index.ts
COPY infra/config.ts /workspace/infra/config.ts
COPY infra/last-preview.txt /workspace/infra/last-preview.txt
COPY infra/package.json /workspace/infra/package.json
COPY infra/Pulumi.yaml /workspace/infra/Pulumi.yaml

WORKDIR /workspace