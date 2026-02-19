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
    apt-get install -y wget unzip && \
    wget -q https://releases.hashicorp.com/terraform/1.5.7/terraform_1.5.7_linux_amd64.zip && \
    unzip terraform_1.5.7_linux_amd64.zip && \
    mv terraform /usr/local/bin/ && \
    rm terraform_1.5.7_linux_amd64.zip && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/terraform-project

COPY data/test-inputs.json /workspace/test-inputs.json
COPY variables.tf /workspace/terraform-project/variables.tf
COPY network.tf /workspace/terraform-project/network.tf
COPY storage.tf /workspace/terraform-project/storage.tf
COPY notifications.tf /workspace/terraform-project/notifications.tf

WORKDIR /workspace