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
    rm -rf /var/lib/apt/lists/*

RUN gem install activerecord sqlite3

RUN mkdir -p /workspace/blog_app/models /workspace/blog_app/data

COPY data/blog.db /workspace/blog_app/blog.db
COPY models/user.rb /workspace/blog_app/models/user.rb
COPY models/post.rb /workspace/blog_app/models/post.rb
COPY models/comment.rb /workspace/blog_app/models/comment.rb
COPY dashboard.rb /workspace/blog_app/dashboard.rb

WORKDIR /workspace/blog_app