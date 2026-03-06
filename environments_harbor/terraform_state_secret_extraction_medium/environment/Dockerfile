FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/terraform_states/project_a \
    /workspace/terraform_states/project_b \
    /workspace/terraform_states/project_c \
    /workspace/terraform_states/legacy \
    /workspace/terraform_states/dev

COPY data/terraform_states/project_a/terraform.tfstate /workspace/terraform_states/project_a/terraform.tfstate
COPY data/terraform_states/project_b/main.tfstate /workspace/terraform_states/project_b/main.tfstate
COPY data/terraform_states/project_c/backend.tfstate /workspace/terraform_states/project_c/backend.tfstate
COPY data/terraform_states/legacy/prod.tfstate /workspace/terraform_states/legacy/prod.tfstate
COPY data/terraform_states/dev/infrastructure.tfstate /workspace/terraform_states/dev/infrastructure.tfstate

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
