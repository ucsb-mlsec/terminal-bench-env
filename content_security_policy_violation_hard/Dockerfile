FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data /config /tmp/solution

COPY data/csp_violation_reports.json /data/csp_violation_reports.json
COPY index.html /index.html
COPY checkout.html /checkout.html
COPY dashboard.html /dashboard.html
COPY profile.html /profile.html
COPY config/current_csp.txt /config/current_csp.txt

WORKDIR /workspace