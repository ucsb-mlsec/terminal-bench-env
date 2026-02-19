FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/test_cases /workspace

COPY data/test_case_1.cnf /workspace/test_cases/test_case_1.cnf
COPY data/test_case_2.cnf /workspace/test_cases/test_case_2.cnf
COPY data/test_case_3.cnf /workspace/test_cases/test_case_3.cnf
COPY data/test_case_4.cnf /workspace/test_cases/test_case_4.cnf
COPY data/test_case_5.cnf /workspace/test_cases/test_case_5.cnf
COPY solver.py /workspace/solver.py

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
