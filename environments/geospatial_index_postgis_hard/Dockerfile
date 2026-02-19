FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install PostgreSQL client and PostGIS
RUN apt-get update && \
    apt-get install -y \
    postgresql-client \
    postgresql-client-16 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y postgresql-client && \
    pip3 install --no-cache-dir --break-system-packages psycopg2-binary && \
    rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /data /tmp

# Copy the database initialization file
COPY data/init_database.sql /data/init_database.sql

# Set working directory
WORKDIR /workspace