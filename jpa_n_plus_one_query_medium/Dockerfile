FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages flask sqlalchemy requests

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/developer/library-api/data

COPY requirements.txt /home/developer/library-api/requirements.txt

RUN pip3 install --no-cache-dir --break-system-packages -r /home/developer/library-api/requirements.txt

COPY models.py /home/developer/library-api/models.py
COPY app.py /home/developer/library-api/app.py
COPY run_and_profile.py /home/developer/library-api/run_and_profile.py
COPY data/database.db /home/developer/library-api/database.db

WORKDIR /home/developer/library-api