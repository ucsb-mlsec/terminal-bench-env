FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /app/api_service /app/data

COPY data/customers.db /app/data/customers.db
COPY api_service/main.py /app/api_service/main.py
COPY api_service/orders.py /app/api_service/orders.py
COPY api_service/products.py /app/api_service/products.py
COPY api_service/reports.py /app/api_service/reports.py
COPY api_service/utils.py /app/api_service/utils.py
COPY api_service/auth.py /app/api_service/auth.py

WORKDIR /app

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
