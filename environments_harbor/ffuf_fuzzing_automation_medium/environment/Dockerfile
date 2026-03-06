FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/www/api_responses /tmp

COPY data/response_x8k2.json /var/www/api_responses/response_x8k2.json
COPY data/response_p9q1.json /var/www/api_responses/response_p9q1.json
COPY data/response_t5w8.json /var/www/api_responses/response_t5w8.json
COPY data/response_d3h6.json /var/www/api_responses/response_d3h6.json
COPY data/response_k2r9.json /var/www/api_responses/response_k2r9.json
COPY data/response_v1b4.json /var/www/api_responses/response_v1b4.json

WORKDIR /var/www/api_responses

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
