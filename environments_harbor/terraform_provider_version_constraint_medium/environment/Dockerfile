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
    apt-get install -y --no-install-recommends \
    wget \
    unzip \
    gnupg \
    software-properties-common && \
    wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/hashicorp.list && \
    apt-get update && \
    apt-get install -y terraform && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y wget curl && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/modules/compute /workspace/modules/networking /workspace/modules/storage

COPY main.tf /workspace/main.tf
COPY versions.tf /workspace/versions.tf
COPY variables.tf /workspace/variables.tf
COPY modules/compute/main.tf /workspace/modules/compute/main.tf
COPY modules/compute/versions.tf /workspace/modules/compute/versions.tf
COPY modules/networking/main.tf /workspace/modules/networking/main.tf
COPY modules/networking/versions.tf /workspace/modules/networking/versions.tf
COPY modules/storage/main.tf /workspace/modules/storage/main.tf
COPY modules/storage/versions.tf /workspace/modules/storage/versions.tf

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
