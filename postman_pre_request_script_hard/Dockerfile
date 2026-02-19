FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/postman-auth/environments /workspace/postman-auth/solution

COPY environments/tenant-alpha.postman_environment.json /workspace/postman-auth/environments/tenant-alpha.postman_environment.json
COPY environments/tenant-beta.postman_environment.json /workspace/postman-auth/environments/tenant-beta.postman_environment.json
COPY oauth-config.json /workspace/postman-auth/oauth-config.json
COPY test-endpoints.txt /workspace/postman-auth/test-endpoints.txt

WORKDIR /workspace/postman-auth