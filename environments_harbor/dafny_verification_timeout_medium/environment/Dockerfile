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
    apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    unzip \
    dotnet-sdk-8.0 && \
    rm -rf /var/lib/apt/lists/*

# Install Dafny 4.0+
RUN wget https://github.com/dafny-lang/dafny/releases/download/v4.4.0/dafny-4.4.0-x64-ubuntu-20.04.zip -O /tmp/dafny.zip && \
    unzip /tmp/dafny.zip -d /opt/dafny && \
    rm /tmp/dafny.zip && \
    chmod +x /opt/dafny/dafny

# Add Dafny to PATH
ENV PATH="/opt/dafny:${PATH}"

# Create workspace directory
RUN mkdir -p /workspace/geometry

# Copy geometry files
COPY geometry/GeometryChecker.dfy /workspace/geometry/GeometryChecker.dfy

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
