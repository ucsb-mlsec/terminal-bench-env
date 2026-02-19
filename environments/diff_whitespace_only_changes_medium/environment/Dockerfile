FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y git && \
    pip3 install --no-cache-dir --break-system-packages requests>=2.28 && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Create necessary directories
RUN mkdir -p /workspace/python_project/.git/refs/heads \
    && mkdir -p /workspace/python_project/.git/objects/aa \
    && mkdir -p /workspace/python_project/.git/objects/bb \
    && mkdir -p /workspace/python_project/services \
    && mkdir -p /workspace/python_project/core

# Copy Git repository structure
COPY .git/HEAD /workspace/python_project/.git/HEAD
COPY .git/refs/heads/main /workspace/python_project/.git/refs/heads/main
COPY .git/refs/heads/feature-refactor /workspace/python_project/.git/refs/heads/feature-refactor
COPY .git/objects/aa/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa /workspace/python_project/.git/objects/aa/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
COPY .git/objects/bb/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb /workspace/python_project/.git/objects/bb/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb

# Copy Python source files
COPY utils.py /workspace/python_project/utils.py
COPY helpers.py /workspace/python_project/helpers.py
COPY models.py /workspace/python_project/models.py
COPY formatters.py /workspace/python_project/formatters.py
COPY services/api.py /workspace/python_project/services/api.py
COPY core/processor.py /workspace/python_project/core/processor.py

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
