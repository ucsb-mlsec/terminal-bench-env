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
    apt-get install -y \
    python3-dev \
    g++ \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y g++ python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pybind11 setuptools wheel

RUN mkdir -p /home/user/stringtools

COPY stringtools/stringtools.cpp /home/user/stringtools/stringtools.cpp
COPY stringtools/setup.py /home/user/stringtools/setup.py
COPY stringtools/pyproject.toml /home/user/stringtools/pyproject.toml
COPY stringtools/test_module.py /home/user/stringtools/test_module.py

WORKDIR /home/user/stringtools