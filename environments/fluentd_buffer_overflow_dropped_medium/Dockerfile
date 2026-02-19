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
    ruby \
    ruby-dev \
    build-essential \
    && gem install fluentd -v 1.16.2 --no-document \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/config /workspace/data /solution /var/log/fluent/buffer

COPY config/fluent.conf /workspace/config/fluent.conf
COPY data/buffer_errors.log /workspace/data/buffer_errors.log

WORKDIR /workspace