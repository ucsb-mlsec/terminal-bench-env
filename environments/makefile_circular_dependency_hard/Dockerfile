FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y make && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/pyproject/modules

COPY modules/alpha.py /workspace/pyproject/modules/alpha.py
COPY modules/beta.py /workspace/pyproject/modules/beta.py
COPY modules/gamma.py /workspace/pyproject/modules/gamma.py
COPY modules/delta.py /workspace/pyproject/modules/delta.py
COPY modules/epsilon.py /workspace/pyproject/modules/epsilon.py
COPY requirements.txt /workspace/pyproject/requirements.txt
COPY Makefile /workspace/pyproject/Makefile

WORKDIR /workspace/pyproject