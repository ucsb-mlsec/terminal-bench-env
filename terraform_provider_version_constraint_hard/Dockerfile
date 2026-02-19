FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Terraform
RUN apt-get update && \
    apt-get install -y wget unzip && \
    wget -q https://releases.hashicorp.com/terraform/1.5.0/terraform_1.5.0_linux_amd64.zip && \
    unzip terraform_1.5.0_linux_amd64.zip && \
    mv terraform /usr/local/bin/ && \
    chmod +x /usr/local/bin/terraform && \
    rm terraform_1.5.0_linux_amd64.zip && \
    rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /workspace/terraform-project/modules/networking && \
    mkdir -p /workspace/terraform-project/modules/compute && \
    mkdir -p /workspace/terraform-project/modules/database && \
    mkdir -p /workspace/terraform-project/modules/monitoring && \
    mkdir -p /workspace/terraform-project/modules/security

# Copy root module files
COPY terraform-project/main.tf /workspace/terraform-project/main.tf
COPY terraform-project/versions.tf /workspace/terraform-project/versions.tf

# Copy networking module files
COPY terraform-project/modules/networking/main.tf /workspace/terraform-project/modules/networking/main.tf
COPY terraform-project/modules/networking/versions.tf /workspace/terraform-project/modules/networking/versions.tf

# Copy compute module files
COPY terraform-project/modules/compute/main.tf /workspace/terraform-project/modules/compute/main.tf
COPY terraform-project/modules/compute/versions.tf /workspace/terraform-project/modules/compute/versions.tf

# Copy database module files
COPY terraform-project/modules/database/main.tf /workspace/terraform-project/modules/database/main.tf
COPY terraform-project/modules/database/versions.tf /workspace/terraform-project/modules/database/versions.tf

# Copy monitoring module files
COPY terraform-project/modules/monitoring/main.tf /workspace/terraform-project/modules/monitoring/main.tf
COPY terraform-project/modules/monitoring/versions.tf /workspace/terraform-project/modules/monitoring/versions.tf

# Copy security module files
COPY terraform-project/modules/security/main.tf /workspace/terraform-project/modules/security/main.tf
COPY terraform-project/modules/security/versions.tf /workspace/terraform-project/modules/security/versions.tf

WORKDIR /workspace