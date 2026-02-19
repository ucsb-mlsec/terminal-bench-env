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
    apt-get install -y openssl && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages cryptography pyopenssl

RUN mkdir -p /data/certificates/web-frontend \
    /data/certificates/api-gateway \
    /data/certificates/payment-service \
    /config/routes \
    /root/solution

COPY data/certificates/web-frontend/current.crt /data/certificates/web-frontend/current.crt
COPY data/certificates/web-frontend/current.key /data/certificates/web-frontend/current.key
COPY data/certificates/web-frontend/new.crt /data/certificates/web-frontend/new.crt
COPY data/certificates/web-frontend/new.key /data/certificates/web-frontend/new.key
COPY data/certificates/web-frontend/ca.crt /data/certificates/web-frontend/ca.crt

COPY data/certificates/api-gateway/current.crt /data/certificates/api-gateway/current.crt
COPY data/certificates/api-gateway/current.key /data/certificates/api-gateway/current.key
COPY data/certificates/api-gateway/new.crt /data/certificates/api-gateway/new.crt
COPY data/certificates/api-gateway/new.key /data/certificates/api-gateway/new.key
COPY data/certificates/api-gateway/ca.crt /data/certificates/api-gateway/ca.crt

COPY data/certificates/payment-service/current.crt /data/certificates/payment-service/current.crt
COPY data/certificates/payment-service/current.key /data/certificates/payment-service/current.key
COPY data/certificates/payment-service/new.crt /data/certificates/payment-service/new.crt
COPY data/certificates/payment-service/new.key /data/certificates/payment-service/new.key
COPY data/certificates/payment-service/ca.crt /data/certificates/payment-service/ca.crt

COPY config/routes/web-frontend.yaml /config/routes/web-frontend.yaml
COPY config/routes/api-gateway.yaml /config/routes/api-gateway.yaml
COPY config/routes/payment-service.yaml /config/routes/payment-service.yaml

WORKDIR /root