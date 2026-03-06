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
    apt-get install -y --no-install-recommends \
    sqlite3 && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages alembic sqlalchemy

RUN mkdir -p /workspace/alembic/versions

COPY alembic.ini /workspace/alembic.ini
COPY alembic/env.py /workspace/alembic/env.py
COPY alembic/script.py.mako /workspace/alembic/script.py.mako
COPY alembic/versions/a1b2c3d4e5f6_add_users_table.py /workspace/alembic/versions/a1b2c3d4e5f6_add_users_table.py
COPY alembic/versions/1a2b3c4d5e6f_add_user_email_index.py /workspace/alembic/versions/1a2b3c4d5e6f_add_user_email_index.py
COPY alembic/versions/9z8y7x6w5v4u_add_user_status_column.py /workspace/alembic/versions/9z8y7x6w5v4u_add_user_status_column.py
COPY production.db /workspace/production.db

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
