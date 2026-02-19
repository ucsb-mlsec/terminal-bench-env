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
    apt-get install -y gawk && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/log/services /home/user/scripts /home/user/output

COPY data/auth.log /var/log/services/auth.log
COPY data/api.log /var/log/services/api.log
COPY data/database.log /var/log/services/database.log
COPY correlate_logs.awk /home/user/scripts/correlate_logs.awk
COPY run_correlation.sh /home/user/scripts/run_correlation.sh

RUN chmod +x /home/user/scripts/run_correlation.sh

WORKDIR /home/user