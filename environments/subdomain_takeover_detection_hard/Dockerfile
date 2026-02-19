FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/audit_data/zone_files /workspace/audit_data/http_responses

COPY data/audit_data/zone_files/example.com.zone /workspace/audit_data/zone_files/example.com.zone
COPY data/audit_data/zone_files/testsite.org.zone /workspace/audit_data/zone_files/testsite.org.zone
COPY data/audit_data/http_responses/blog.example.com.txt /workspace/audit_data/http_responses/blog.example.com.txt
COPY data/audit_data/http_responses/dev.example.com.txt /workspace/audit_data/http_responses/dev.example.com.txt
COPY data/audit_data/http_responses/staging.example.com.txt /workspace/audit_data/http_responses/staging.example.com.txt
COPY data/audit_data/http_responses/api.example.com.txt /workspace/audit_data/http_responses/api.example.com.txt
COPY data/audit_data/http_responses/docs.example.com.txt /workspace/audit_data/http_responses/docs.example.com.txt
COPY data/audit_data/http_responses/cdn.testsite.org.txt /workspace/audit_data/http_responses/cdn.testsite.org.txt
COPY data/audit_data/http_responses/portal.testsite.org.txt /workspace/audit_data/http_responses/portal.testsite.org.txt
COPY data/audit_data/http_responses/old.testsite.org.txt /workspace/audit_data/http_responses/old.testsite.org.txt
COPY data/audit_data/http_responses/support.testsite.org.txt /workspace/audit_data/http_responses/support.testsite.org.txt
COPY data/audit_data/service_signatures.txt /workspace/audit_data/service_signatures.txt

WORKDIR /workspace