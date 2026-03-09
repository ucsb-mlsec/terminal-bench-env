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
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/data /workspace/tests /workspace/services /workspace/monitoring /home/solution \
    /workspace/output/k6_load_test_thresholds_hard/files/tests \
    /workspace/output/k6_load_test_thresholds_hard/files/services \
    /workspace/output/k6_load_test_thresholds_hard/files/monitoring

COPY data/SLA_REQUIREMENTS.md /workspace/data/SLA_REQUIREMENTS.md
COPY tests/user-service.test.js /workspace/tests/user-service.test.js
COPY tests/payment-service.test.js /workspace/tests/payment-service.test.js
COPY tests/inventory-service.test.js /workspace/tests/inventory-service.test.js
COPY services/api-gateway.test.js /workspace/services/api-gateway.test.js
COPY monitoring/metrics-test.js /workspace/monitoring/metrics-test.js

COPY tests/user-service.test.js /workspace/output/k6_load_test_thresholds_hard/files/tests/user-service.test.js
COPY tests/payment-service.test.js /workspace/output/k6_load_test_thresholds_hard/files/tests/payment-service.test.js
COPY tests/inventory-service.test.js /workspace/output/k6_load_test_thresholds_hard/files/tests/inventory-service.test.js
COPY services/api-gateway.test.js /workspace/output/k6_load_test_thresholds_hard/files/services/api-gateway.test.js
COPY monitoring/metrics-test.js /workspace/output/k6_load_test_thresholds_hard/files/monitoring/metrics-test.js

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
