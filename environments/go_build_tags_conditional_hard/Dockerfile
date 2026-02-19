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
    apt-get install -y golang-go && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y golang && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user/dataservice /tmp/bin /tmp/solution

COPY dataservice/main.go /home/user/dataservice/main.go
COPY dataservice/storage_cloud.go /home/user/dataservice/storage_cloud.go
COPY dataservice/storage_edge.go /home/user/dataservice/storage_edge.go
COPY dataservice/storage_default.go /home/user/dataservice/storage_default.go
COPY dataservice/storage.go /home/user/dataservice/storage.go
COPY dataservice/config.go /home/user/dataservice/config.go
COPY dataservice/go.mod /home/user/dataservice/go.mod

WORKDIR /home/user/dataservice