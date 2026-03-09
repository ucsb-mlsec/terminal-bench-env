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
    apt-get install -y php-cli php-json && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y php && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/app/models /workspace/app /workspace/solution

COPY data/spec.json /workspace/app/spec.json
COPY models/Customer.php /workspace/app/models/Customer.php
COPY models/Invoice.php /workspace/app/models/Invoice.php
COPY models/Payment.php /workspace/app/models/Payment.php
COPY models/Subscription.php /workspace/app/models/Subscription.php
COPY models/Transaction.php /workspace/app/models/Transaction.php

WORKDIR /workspace/app

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
