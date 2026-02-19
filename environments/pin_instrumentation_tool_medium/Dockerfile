FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Create necessary directories
RUN mkdir -p /workspace /opt/pin

# Install necessary dependencies for C compilation and Intel Pin
RUN apt-get update && \
    apt-get install -y \
    gcc \
    g++ \
    make \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y build-essential g++ make && \
    rm -rf /var/lib/apt/lists/*

# Download and install Intel Pin
RUN cd /opt && \
    wget -q https://software.intel.com/sites/landingpage/pintool/downloads/pin-3.30-98830-g1d7b601b3-gcc-linux.tar.gz && \
    tar -xzf pin-3.30-98830-g1d7b601b3-gcc-linux.tar.gz && \
    mv pin-3.30-98830-g1d7b601b3-gcc-linux pin && \
    rm pin-3.30-98830-g1d7b601b3-gcc-linux.tar.gz

# Copy target program source file
COPY target_program.c /workspace/target_program.c

WORKDIR /workspace