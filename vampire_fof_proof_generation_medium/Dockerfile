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
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages python-sat

RUN pip3 install --no-cache-dir --break-system-packages pysat

RUN mkdir -p /workspace/properties

COPY properties/property_1.cnf /workspace/properties/property_1.cnf
COPY properties/property_2.cnf /workspace/properties/property_2.cnf
COPY properties/property_3.cnf /workspace/properties/property_3.cnf
COPY properties/property_4.cnf /workspace/properties/property_4.cnf
COPY properties/property_5.cnf /workspace/properties/property_5.cnf

WORKDIR /workspace