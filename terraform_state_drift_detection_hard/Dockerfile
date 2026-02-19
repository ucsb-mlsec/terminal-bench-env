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
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

RUN wget -q https://releases.hashicorp.com/terraform/1.5.0/terraform_1.5.0_linux_amd64.zip && \
    unzip terraform_1.5.0_linux_amd64.zip && \
    mv terraform /usr/local/bin/ && \
    rm terraform_1.5.0_linux_amd64.zip

RUN mkdir -p /workspace/config /home/candidate

COPY main.tf /workspace/main.tf
COPY terraform.tfstate /workspace/terraform.tfstate
COPY config/security_policy.yaml /workspace/config/security_policy.yaml
COPY config/drift_categories.yaml /workspace/config/drift_categories.yaml

WORKDIR /workspace