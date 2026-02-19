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

RUN pip3 install --no-cache-dir --break-system-packages django

RUN mkdir -p /app/myproject /app/dashboard /app/api /app/legacy /app/widgets

COPY myproject/__init__.py /app/myproject/__init__.py
COPY myproject/settings.py /app/myproject/settings.py
COPY myproject/urls.py /app/myproject/urls.py
COPY dashboard/__init__.py /app/dashboard/__init__.py
COPY dashboard/views.py /app/dashboard/views.py
COPY dashboard/urls.py /app/dashboard/urls.py
COPY api/__init__.py /app/api/__init__.py
COPY api/views.py /app/api/views.py
COPY api/urls.py /app/api/urls.py
COPY legacy/__init__.py /app/legacy/__init__.py
COPY legacy/views.py /app/legacy/views.py
COPY legacy/urls.py /app/legacy/urls.py
COPY widgets/__init__.py /app/widgets/__init__.py
COPY widgets/views.py /app/widgets/views.py
COPY widgets/urls.py /app/widgets/urls.py

WORKDIR /app