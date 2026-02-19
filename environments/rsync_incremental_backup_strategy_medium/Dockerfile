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
    apt-get install -y rsync && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/projects/project1/data \
             /var/projects/project2 \
             /var/projects/project3 \
             /var/projects/shared \
             /var/projects/docs \
             /var/projects/logs \
             /backup/storage \
             /root

COPY var/projects/project1/main.py /var/projects/project1/main.py
COPY var/projects/project1/config.json /var/projects/project1/config.json
COPY var/projects/project1/README.md /var/projects/project1/README.md
COPY var/projects/project1/data/users.json /var/projects/project1/data/users.json
COPY var/projects/project2/app.js /var/projects/project2/app.js
COPY var/projects/project2/package.json /var/projects/project2/package.json
COPY var/projects/project2/data.csv /var/projects/project2/data.csv
COPY var/projects/project3/deploy.sh /var/projects/project3/deploy.sh
COPY var/projects/project3/requirements.txt /var/projects/project3/requirements.txt
COPY var/projects/project3/settings.ini /var/projects/project3/settings.ini
COPY var/projects/shared/utils.py /var/projects/shared/utils.py
COPY var/projects/shared/constants.py /var/projects/shared/constants.py
COPY var/projects/docs/architecture.md /var/projects/docs/architecture.md
COPY var/projects/logs/app.log /var/projects/logs/app.log
COPY var/projects/.gitignore /var/projects/.gitignore

RUN chmod +x /var/projects/project3/deploy.sh

WORKDIR /root