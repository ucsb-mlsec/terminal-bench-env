FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip3 install --no-cache-dir --break-system-packages locust pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/locust_cluster

# Create locustfile.py
RUN echo 'from locust import HttpUser, task, between\n\
\n\
class WebsiteUser(HttpUser):\n\
    wait_time = between(1, 5)\n\
\n\
    @task\n\
    def load_page(self):\n\
        self.client.get("/")' > /workspace/locust_cluster/locustfile.py

# Create master.conf with bug (wrong port)
RUN echo 'locustfile = locustfile.py\n\
master = true\n\
master-bind-host = 0.0.0.0\n\
master-bind-port = 5558\n\
web-host = 0.0.0.0\n\
web-port = 8089' > /workspace/locust_cluster/master.conf

# Create worker1.conf with bug (wrong host)
RUN echo 'locustfile = locustfile.py\n\
worker = true\n\
master-host = 127.0.0.1\n\
master-port = 5557' > /workspace/locust_cluster/worker1.conf

# Create worker2.conf with bug (typo in parameter name)
RUN echo 'locustfile = locustfile.py\n\
worker = true\n\
master_host = localhost\n\
master-port = 5557' > /workspace/locust_cluster/worker2.conf

# Create requirements.txt
RUN echo 'locust>=2.0.0' > /workspace/locust_cluster/requirements.txt

WORKDIR /workspace