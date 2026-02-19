FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/job_definitions

COPY data/job_definitions/job001.job /workspace/job_definitions/job001.job
COPY data/job_definitions/job002.job /workspace/job_definitions/job002.job
COPY data/job_definitions/job003.job /workspace/job_definitions/job003.job
COPY data/job_definitions/job004.job /workspace/job_definitions/job004.job
COPY data/job_definitions/job005.job /workspace/job_definitions/job005.job
COPY data/job_definitions/job006.job /workspace/job_definitions/job006.job
COPY data/job_definitions/job007.job /workspace/job_definitions/job007.job
COPY data/job_definitions/job008.job /workspace/job_definitions/job008.job
COPY data/job_definitions/job009.job /workspace/job_definitions/job009.job
COPY data/job_definitions/job010.job /workspace/job_definitions/job010.job
COPY data/job_definitions/job011.job /workspace/job_definitions/job011.job
COPY data/job_definitions/job012.job /workspace/job_definitions/job012.job
COPY data/job_definitions/job013.job /workspace/job_definitions/job013.job
COPY data/job_definitions/job014.job /workspace/job_definitions/job014.job
COPY data/job_definitions/job015.job /workspace/job_definitions/job015.job

WORKDIR /workspace