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
    postgresql-client \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages psycopg2-binary

RUN pip3 install --no-cache-dir --break-system-packages \
    psycopg2-binary \
    phonenumbers \
    email-validator

RUN mkdir -p /home/user /config /data

COPY config/census_config.json /config/census_config.json
COPY config/salesforce_credentials.json /config/salesforce_credentials.json
COPY config/database_config.json /config/database_config.json
COPY data/sample_customer_data.sql /data/sample_customer_data.sql

WORKDIR /home/user