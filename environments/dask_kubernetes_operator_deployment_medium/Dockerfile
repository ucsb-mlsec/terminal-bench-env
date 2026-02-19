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
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages pyyaml

RUN pip3 install --no-cache-dir --break-system-packages \
    pyyaml \
    kubernetes

RUN mkdir -p /config /tmp

COPY config/dask-cluster-scheduler.yaml /config/dask-cluster-scheduler.yaml
COPY config/dask-cluster-workers.yaml /config/dask-cluster-workers.yaml
COPY config/dask-scheduler-service.yaml /config/dask-scheduler-service.yaml
COPY config/dask-namespace.yaml /config/dask-namespace.yaml

WORKDIR /workspace