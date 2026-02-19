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
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

RUN mkdir -p /workspace/microservice-a \
    /workspace/microservice-b \
    /workspace/microservice-c \
    /workspace/microservice-d \
    /workspace/microservice-e \
    /workspace/microservice-f

COPY microservice-a/database-secret.yaml /workspace/microservice-a/database-secret.yaml
COPY microservice-a/redis-secret.yaml /workspace/microservice-a/redis-secret.yaml
COPY microservice-a/api-token.yaml /workspace/microservice-a/api-token.yaml
COPY microservice-b/postgres-secret.yaml /workspace/microservice-b/postgres-secret.yaml
COPY microservice-b/oauth-secret.yaml /workspace/microservice-b/oauth-secret.yaml
COPY microservice-b/ssl-cert.yaml /workspace/microservice-b/ssl-cert.yaml
COPY microservice-c/mongodb-secret.yaml /workspace/microservice-c/mongodb-secret.yaml
COPY microservice-c/jwt-secret.yaml /workspace/microservice-c/jwt-secret.yaml
COPY microservice-c/smtp-config.yaml /workspace/microservice-c/smtp-config.yaml
COPY microservice-d/kafka-secret.yaml /workspace/microservice-d/kafka-secret.yaml
COPY microservice-d/s3-credentials.yaml /workspace/microservice-d/s3-credentials.yaml
COPY microservice-e/ldap-secret.yaml /workspace/microservice-e/ldap-secret.yaml
COPY microservice-e/rabbitmq-secret.yaml /workspace/microservice-e/rabbitmq-secret.yaml
COPY microservice-f/grafana-secret.yaml /workspace/microservice-f/grafana-secret.yaml
COPY microservice-f/docker-registry.yaml /workspace/microservice-f/docker-registry.yaml

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
