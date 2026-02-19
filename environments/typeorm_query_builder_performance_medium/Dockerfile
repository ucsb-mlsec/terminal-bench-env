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
    apt-get install -y nodejs npm sqlite3 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/product-api/entities /workspace/data

COPY product-api/package.json /workspace/product-api/package.json
COPY product-api/entities/Product.js /workspace/product-api/entities/Product.js
COPY product-api/entities/Category.js /workspace/product-api/entities/Category.js
COPY product-api/server.js /workspace/product-api/server.js
COPY product-api/test-performance.js /workspace/product-api/test-performance.js
COPY data/products.db /workspace/data/products.db

WORKDIR /workspace/product-api

RUN npm install

WORKDIR /workspace