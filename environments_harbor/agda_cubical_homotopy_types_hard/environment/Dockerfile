FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Agda and dependencies
RUN apt-get update && \
    apt-get install -y \
    agda \
    agda-stdlib \
    wget \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create directories for Agda libraries and configuration
RUN mkdir -p /usr/local/lib/agda/standard-library && \
    mkdir -p /usr/local/lib/agda/cubical && \
    mkdir -p ~/.agda

# Install Agda standard library
RUN wget -q https://github.com/agda/agda-stdlib/archive/v2.0.tar.gz -O /tmp/agda-stdlib.tar.gz && \
    tar -xzf /tmp/agda-stdlib.tar.gz -C /tmp && \
    cp -r /tmp/agda-stdlib-2.0/* /usr/local/lib/agda/standard-library/ && \
    rm -rf /tmp/agda-stdlib.tar.gz /tmp/agda-stdlib-2.0

# Install Cubical Agda library
RUN wget -q https://github.com/agda/cubical/archive/v0.7.tar.gz -O /tmp/cubical.tar.gz && \
    tar -xzf /tmp/cubical.tar.gz -C /tmp && \
    cp -r /tmp/cubical-0.7/* /usr/local/lib/agda/cubical/ && \
    rm -rf /tmp/cubical.tar.gz /tmp/cubical-0.7

# Configure Agda libraries
RUN echo "/usr/local/lib/agda/standard-library/standard-library.agda-lib" > ~/.agda/libraries && \
    echo "/usr/local/lib/agda/cubical/cubical.agda-lib" >> ~/.agda/libraries && \
    echo "standard-library" > ~/.agda/defaults && \
    echo "cubical" >> ~/.agda/defaults

# Create working directory
RUN mkdir -p /workspace

# Copy the Agda module
COPY PathProperties.agda /workspace/PathProperties.agda

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
