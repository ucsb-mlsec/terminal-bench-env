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
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/arminbiere/kissat/archive/refs/tags/sc2022-light.tar.gz && \
    tar -xzf sc2022-light.tar.gz && \
    cd kissat-sc2022-light && \
    ./configure && \
    make && \
    cp build/kissat /usr/local/bin/ && \
    cd .. && \
    rm -rf kissat-sc2022-light sc2022-light.tar.gz

RUN mkdir -p /workspace/bve_tuning/problems

COPY data/problems/verify_01.cnf /workspace/bve_tuning/problems/verify_01.cnf
COPY data/problems/verify_02.cnf /workspace/bve_tuning/problems/verify_02.cnf
COPY data/problems/verify_03.cnf /workspace/bve_tuning/problems/verify_03.cnf

WORKDIR /workspace/bve_tuning