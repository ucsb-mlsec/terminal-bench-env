FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages flask-sqlalchemy

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/flask_app/alembic/versions

RUN pip3 install --no-cache-dir --break-system-packages flask alembic sqlalchemy

COPY flask_app/requirements.txt /workspace/flask_app/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /workspace/flask_app/requirements.txt

COPY flask_app/app.py /workspace/flask_app/app.py
COPY flask_app/alembic.ini /workspace/flask_app/alembic.ini
COPY flask_app/alembic/env.py /workspace/flask_app/alembic/env.py
COPY flask_app/alembic/script.py.mako /workspace/flask_app/alembic/script.py.mako
COPY flask_app/alembic/versions/a1b2c3d4e5f6_initial_migration.py /workspace/flask_app/alembic/versions/a1b2c3d4e5f6_initial_migration.py
COPY flask_app/alembic/versions/b2c3d4e5f6g7_add_posts_table.py /workspace/flask_app/alembic/versions/b2c3d4e5f6g7_add_posts_table.py
COPY flask_app/alembic/versions/c3d4e5f6g7h8_add_comments_table.py /workspace/flask_app/alembic/versions/c3d4e5f6g7h8_add_comments_table.py

WORKDIR /workspace/flask_app