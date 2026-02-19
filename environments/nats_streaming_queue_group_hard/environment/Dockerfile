FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Go and necessary dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    golang-go \
    git \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y golang && \
    rm -rf /var/lib/apt/lists/*

# Set Go environment variables
ENV GOPATH=/go
ENV PATH=$PATH:/usr/local/go/bin:$GOPATH/bin

# Create workspace directory
RUN mkdir -p /workspace

# Copy generated files to the container
COPY subscriber.go /workspace/subscriber.go
COPY BUGS.txt /workspace/BUGS.txt

# Set working directory
WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
