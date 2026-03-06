FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Lean 4 dependencies
RUN apt-get update && \
    apt-get install -y curl git && \
    rm -rf /var/lib/apt/lists/*

# Install elan (Lean version manager)
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain none && \
    echo 'export PATH="$HOME/.elan/bin:$PATH"' >> ~/.bashrc

ENV PATH="/root/.elan/bin:$PATH"

# Create workspace directory
RUN mkdir -p /workspace/lean_project

# Copy Lean project files
COPY lean_project/LeanProof.lean /workspace/lean_project/LeanProof.lean
COPY lean_project/lakefile.lean /workspace/lean_project/lakefile.lean
COPY lean_project/lean-toolchain /workspace/lean_project/lean-toolchain

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
