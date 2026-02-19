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
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages jq

RUN mkdir -p /workspace/terraform_states

COPY data/terraform_states/dev.tfstate /workspace/terraform_states/dev.tfstate
COPY data/terraform_states/staging.tfstate /workspace/terraform_states/staging.tfstate
COPY data/terraform_states/prod.tfstate /workspace/terraform_states/prod.tfstate
COPY data/terraform_states/legacy.tfstate /workspace/terraform_states/legacy.tfstate
COPY data/terraform_states/test.tfstate /workspace/terraform_states/test.tfstate

WORKDIR /workspace