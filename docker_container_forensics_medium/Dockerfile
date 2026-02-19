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
    apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages flask

RUN mkdir -p /forensics /solution /app /tmp /var/log /home/appuser /forensics/container_snapshot/app /forensics/container_snapshot/tmp /forensics/container_snapshot/var/log /forensics/container_snapshot/home/appuser /forensics/container_snapshot/app/__pycache__

COPY forensics/clean_app.py /forensics/clean_app.py
COPY forensics/container_snapshot/app/app.py /forensics/container_snapshot/app/app.py
COPY forensics/container_snapshot/tmp/.backdoor.sh /forensics/container_snapshot/tmp/.backdoor.sh
COPY forensics/container_snapshot/app/__pycache__/app.cpython-38.pyc /forensics/container_snapshot/app/__pycache__/app.cpython-38.pyc
COPY forensics/container_snapshot/var/log/flask.log /forensics/container_snapshot/var/log/flask.log
COPY forensics/container_snapshot/home/appuser/.bash_history /forensics/container_snapshot/home/appuser/.bash_history
COPY forensics/container_snapshot/tmp/system_update.log /forensics/container_snapshot/tmp/system_update.log

WORKDIR /forensics