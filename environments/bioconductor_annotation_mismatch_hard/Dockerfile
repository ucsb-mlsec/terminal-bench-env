FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/data /workspace/solution

COPY data/expression_data.csv /workspace/data/expression_data.csv
COPY data/reference_mappings.csv /workspace/data/reference_mappings.csv
COPY data/pathway_database.txt /workspace/data/pathway_database.txt

RUN pip3 install --no-cache-dir --break-system-packages pandas

WORKDIR /workspace