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
    apt-get install -y nodejs npm sqlite3 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/migrations /workspace/seeds

COPY knexfile.js /workspace/knexfile.js
COPY package.json /workspace/package.json
COPY migrations/20230101000001_create_users.js /workspace/migrations/20230101000001_create_users.js
COPY migrations/20230101000002_create_posts.js /workspace/migrations/20230101000002_create_posts.js
COPY migrations/20230101000003_create_comments.js /workspace/migrations/20230101000003_create_comments.js
COPY seeds/01_users.js /workspace/seeds/01_users.js
COPY seeds/02_posts.js /workspace/seeds/02_posts.js
COPY seeds/03_comments.js /workspace/seeds/03_comments.js

WORKDIR /workspace

RUN npm install