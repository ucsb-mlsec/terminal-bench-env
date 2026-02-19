FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/pipeline/data /workspace/solution

COPY pipeline/process_stage1.sh /workspace/pipeline/process_stage1.sh
COPY pipeline/process_stage2.sh /workspace/pipeline/process_stage2.sh
COPY pipeline/process_stage3.sh /workspace/pipeline/process_stage3.sh
COPY pipeline/process_stage4.sh /workspace/pipeline/process_stage4.sh
COPY pipeline/run_pipeline.sh /workspace/pipeline/run_pipeline.sh
COPY pipeline/data/raw_data.txt /workspace/pipeline/data/raw_data.txt

RUN chmod +x /workspace/pipeline/process_stage1.sh && \
    chmod +x /workspace/pipeline/process_stage2.sh && \
    chmod +x /workspace/pipeline/process_stage3.sh && \
    chmod +x /workspace/pipeline/process_stage4.sh && \
    chmod +x /workspace/pipeline/run_pipeline.sh

WORKDIR /workspace/pipeline