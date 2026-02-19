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

RUN mkdir -p /app/models /app /solution

COPY data/spec.json /app/spec.json
COPY models/Customer.php /app/models/Customer.php
COPY models/Invoice.php /app/models/Invoice.php
COPY models/Payment.php /app/models/Payment.php
COPY models/Subscription.php /app/models/Subscription.php
COPY models/Transaction.php /app/models/Transaction.php

WORKDIR /app