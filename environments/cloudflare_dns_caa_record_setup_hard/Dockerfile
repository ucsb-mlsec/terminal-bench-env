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
    jq \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages jsonschema

RUN mkdir -p /data /config /tmp

COPY data/security_policy.json /data/security_policy.json
COPY data/certificate_issuance_logs.json /data/certificate_issuance_logs.json
COPY data/domain_ownership_verification.json /data/domain_ownership_verification.json
COPY config/cloudflare_api_format.json /config/cloudflare_api_format.json

WORKDIR /workspace