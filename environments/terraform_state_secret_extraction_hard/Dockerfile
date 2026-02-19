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
    gzip \
    tar \
    file && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/terraform-states/production && \
    mkdir -p /workspace/terraform-states/staging && \
    mkdir -p /workspace/terraform-states/development

COPY data/terraform-states/production/terraform.tfstate /workspace/terraform-states/production/terraform.tfstate
COPY data/terraform-states/staging/terraform.tfstate.backup /workspace/terraform-states/staging/terraform.tfstate.backup
COPY data/terraform-states/development/dev.tfstate.gz /workspace/terraform-states/development/dev.tfstate.gz
COPY data/terraform-states/production/backend.tfstate /workspace/terraform-states/production/backend.tfstate
COPY data/terraform-states/staging/encrypted.tfstate /workspace/terraform-states/staging/encrypted.tfstate
COPY data/terraform-states/development/partial.tfstate /workspace/terraform-states/development/partial.tfstate
COPY data/terraform-states/production/modules.tfstate /workspace/terraform-states/production/modules.tfstate
COPY data/terraform-states/staging/mixed.tfstate.tar.gz /workspace/terraform-states/staging/mixed.tfstate.tar.gz
COPY data/terraform-states/development/.terraform.tfstate.lock.info /workspace/terraform-states/development/.terraform.tfstate.lock.info
COPY data/terraform-states/staging/workspace-staging.tfstate /workspace/terraform-states/staging/workspace-staging.tfstate
COPY data/terraform-states/development/test-infra.tfstate /workspace/terraform-states/development/test-infra.tfstate

WORKDIR /workspace