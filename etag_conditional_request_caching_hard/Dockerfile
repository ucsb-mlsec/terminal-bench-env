FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip sqlite3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Flask and requests
RUN pip3 install --no-cache-dir --break-system-packages flask requests

# Create directories
RUN mkdir -p /home/api/product_service /home/api

# Copy application files
COPY app.py /home/api/product_service/app.py
COPY requirements.txt /home/api/product_service/requirements.txt
COPY test_client.py /home/api/product_service/test_client.py

# Set working directory
WORKDIR /home/api/product_service

# Create and populate database
RUN python3 -c "import sqlite3; conn = sqlite3.connect('database.db'); c = conn.cursor(); c.execute('CREATE TABLE IF NOT EXISTS products (id INTEGER PRIMARY KEY, name TEXT, price REAL, description TEXT, last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP)'); c.execute('INSERT INTO products (id, name, price, description) VALUES (1, \"Widget\", 19.99, \"A useful widget\"), (2, \"Gadget\", 29.99, \"A fancy gadget\"), (3, \"Doohickey\", 9.99, \"A small doohickey\")'); conn.commit(); conn.close()"