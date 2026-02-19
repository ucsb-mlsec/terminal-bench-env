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
    apt-get install -y \
    gawk \
    bc \
    jq && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y bc && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/finance/data \
    /opt/finance/scripts \
    /opt/finance/validation \
    /opt/finance/solution

COPY data/visa_transactions.csv /opt/finance/data/visa_transactions.csv
COPY data/mastercard_transactions.csv /opt/finance/data/mastercard_transactions.csv
COPY data/ach_transfers.csv /opt/finance/data/ach_transfers.csv
COPY data/adjustments.csv /opt/finance/data/adjustments.csv
COPY data/process_transactions.awk /opt/finance/scripts/process_transactions.awk
COPY data/aggregate_daily.awk /opt/finance/scripts/aggregate_daily.awk
COPY data/expected_totals.txt /opt/finance/validation/expected_totals.txt

WORKDIR /opt/finance