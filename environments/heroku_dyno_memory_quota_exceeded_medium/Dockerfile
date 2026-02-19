FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages matplotlib

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /app /app/config /app/data /home/agent

COPY app.py /app/app.py
COPY config/heroku.yml /app/config/heroku.yml
COPY config/app_config.py /app/config/app_config.py
COPY requirements.txt /app/requirements.txt
COPY data/heroku_logs.txt /app/data/heroku_logs.txt
COPY Procfile /app/Procfile

WORKDIR /app

RUN pip3 install --no-cache-dir --break-system-packages flask