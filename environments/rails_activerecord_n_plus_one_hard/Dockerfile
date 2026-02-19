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
    apt-get install -y ruby ruby-dev build-essential libsqlite3-dev && \
    gem install activerecord sqlite3 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app /app/data /solution

COPY data/blog.db /app/data/blog.db
COPY post.rb /app/post.rb
COPY comment.rb /app/comment.rb
COPY author.rb /app/author.rb
COPY generate_report.rb /app/generate_report.rb

WORKDIR /app