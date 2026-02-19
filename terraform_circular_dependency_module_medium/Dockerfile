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
    wget https://releases.hashicorp.com/terraform/1.5.7/terraform_1.5.7_linux_amd64.zip && \
    unzip terraform_1.5.7_linux_amd64.zip && \
    mv terraform /usr/local/bin/ && \
    chmod +x /usr/local/bin/terraform && \
    rm terraform_1.5.7_linux_amd64.zip && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /terraform-project/modules/network && \
    mkdir -p /terraform-project/modules/storage && \
    mkdir -p /terraform-project/root && \
    mkdir -p /solution

COPY terraform-project/modules/network/main.tf /terraform-project/modules/network/main.tf
COPY terraform-project/modules/network/variables.tf /terraform-project/modules/network/variables.tf
COPY terraform-project/modules/network/outputs.tf /terraform-project/modules/network/outputs.tf
COPY terraform-project/modules/storage/main.tf /terraform-project/modules/storage/main.tf
COPY terraform-project/modules/storage/variables.tf /terraform-project/modules/storage/variables.tf
COPY terraform-project/modules/storage/outputs.tf /terraform-project/modules/storage/outputs.tf
COPY terraform-project/root/main.tf /terraform-project/root/main.tf
COPY terraform-project/root/variables.tf /terraform-project/root/variables.tf
COPY terraform-project/root/outputs.tf /terraform-project/root/outputs.tf

WORKDIR /terraform-project/root