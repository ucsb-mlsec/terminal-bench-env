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
    ca-certificates && \
    wget -O /tmp/cvc5.tar.gz https://github.com/cvc5/cvc5/releases/download/cvc5-1.0.8/cvc5-Linux && \
    chmod +x /tmp/cvc5.tar.gz && \
    mv /tmp/cvc5.tar.gz /usr/local/bin/cvc5 && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y cvc5 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace

COPY password_policies.smt2 /workspace/password_policies.smt2

WORKDIR /workspace