FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/priors /workspace/solution

COPY data/priors/prior_001.txt /workspace/priors/prior_001.txt
COPY data/priors/prior_002.txt /workspace/priors/prior_002.txt
COPY data/priors/prior_003.txt /workspace/priors/prior_003.txt
COPY data/priors/prior_004.txt /workspace/priors/prior_004.txt
COPY data/priors/prior_005.txt /workspace/priors/prior_005.txt
COPY data/priors/prior_006.txt /workspace/priors/prior_006.txt
COPY data/priors/prior_007.txt /workspace/priors/prior_007.txt
COPY data/priors/prior_008.txt /workspace/priors/prior_008.txt
COPY data/priors/prior_009.txt /workspace/priors/prior_009.txt
COPY data/priors/prior_010.txt /workspace/priors/prior_010.txt
COPY data/priors/prior_011.txt /workspace/priors/prior_011.txt
COPY data/priors/prior_012.txt /workspace/priors/prior_012.txt
COPY data/priors/prior_013.txt /workspace/priors/prior_013.txt
COPY data/priors/prior_014.txt /workspace/priors/prior_014.txt
COPY data/priors/prior_015.txt /workspace/priors/prior_015.txt

WORKDIR /workspace