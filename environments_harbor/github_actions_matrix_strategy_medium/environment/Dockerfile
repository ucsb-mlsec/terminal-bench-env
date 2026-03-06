FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install python3-pip, git and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip git && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Create directory structure
RUN mkdir -p /workspace/.github/workflows

WORKDIR /workspace

# Copy the required files into the container
COPY ci.yml /workspace/.github/workflows/ci.yml
COPY workspace_requirements.txt /workspace/requirements.txt

RUN cp /workspace/.github/workflows/ci.yml /workspace/fixed-ci.yml

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
