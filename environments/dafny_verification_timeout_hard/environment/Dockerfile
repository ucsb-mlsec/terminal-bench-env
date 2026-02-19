FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Dafny dependencies
RUN apt-get update && \
    apt-get install -y wget unzip dotnet-sdk-8.0 && \
    rm -rf /var/lib/apt/lists/*

# Install Dafny 4.2.0
RUN wget https://github.com/dafny-lang/dafny/releases/download/v4.2.0/dafny-4.2.0-x64-ubuntu-20.04.zip -O /tmp/dafny.zip && \
    unzip /tmp/dafny.zip -d /opt/ && \
    rm /tmp/dafny.zip && \
    chmod +x /opt/dafny/dafny

# Add Dafny to PATH
ENV PATH="/opt/dafny:${PATH}"

# Create workspace directory
RUN mkdir -p /workspace

# Copy the Dafny specification file
COPY ListReversal.dfy /workspace/ListReversal.dfy

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
