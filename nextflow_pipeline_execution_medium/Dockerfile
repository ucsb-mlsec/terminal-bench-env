FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Nextflow and Java runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    openjdk-17-jre-headless && \
    curl -s https://get.nextflow.io | bash && \
    mv nextflow /usr/local/bin/ && \
    chmod +x /usr/local/bin/nextflow && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y default-jre-headless procps && \
    rm -rf /var/lib/apt/lists/*

# Create required directories
RUN mkdir -p /workspace/input_data /solution

# Copy pipeline files to expected locations
COPY qc_pipeline.nf /workspace/qc_pipeline.nf
COPY config/nextflow.config /workspace/nextflow.config
COPY data/sample1.fastq /workspace/input_data/sample1.fastq
COPY data/sample2.fastq /workspace/input_data/sample2.fastq
COPY data/sample3.fastq /workspace/input_data/sample3.fastq

WORKDIR /workspace