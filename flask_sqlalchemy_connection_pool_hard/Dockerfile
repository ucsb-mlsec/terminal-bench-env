FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y curl && \
    pip3 install --no-cache-dir --break-system-packages flask-sqlalchemy && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /app /solution

RUN pip3 install --no-cache-dir --break-system-packages flask sqlalchemy requests

COPY app.py /app/app.py
COPY routes.py /app/routes.py
COPY run_load_test.sh /app/run_load_test.sh
COPY requirements.txt /app/requirements.txt

RUN chmod +x /app/run_load_test.sh

WORKDIR /app