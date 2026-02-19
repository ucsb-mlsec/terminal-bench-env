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
    apt-get install -y wget unzip curl gnupg software-properties-common && \
    wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/hashicorp.list && \
    apt-get update && \
    apt-get install -y terraform && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/dev /workspace/staging /workspace/prod \
    /workspace/modules/networking /workspace/modules/application \
    /workspace/modules/database /workspace/modules/monitoring \
    /workspace/modules/logging

COPY dev/main.tf /workspace/dev/main.tf
COPY staging/main.tf /workspace/staging/main.tf
COPY prod/main.tf /workspace/prod/main.tf
COPY modules/networking/main.tf /workspace/modules/networking/main.tf
COPY modules/networking/variables.tf /workspace/modules/networking/variables.tf
COPY modules/networking/outputs.tf /workspace/modules/networking/outputs.tf
COPY modules/application/main.tf /workspace/modules/application/main.tf
COPY modules/application/variables.tf /workspace/modules/application/variables.tf
COPY modules/database/main.tf /workspace/modules/database/main.tf
COPY modules/database/variables.tf /workspace/modules/database/variables.tf
COPY modules/monitoring/main.tf /workspace/modules/monitoring/main.tf
COPY modules/monitoring/variables.tf /workspace/modules/monitoring/variables.tf
COPY modules/logging/main.tf /workspace/modules/logging/main.tf
COPY modules/logging/variables.tf /workspace/modules/logging/variables.tf

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
