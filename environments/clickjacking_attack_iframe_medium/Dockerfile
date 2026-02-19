FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /investigation /solution

COPY investigation/page1.html /investigation/page1.html
COPY investigation/page2.html /investigation/page2.html
COPY investigation/page3.html /investigation/page3.html
COPY investigation/legitimate1.html /investigation/legitimate1.html
COPY investigation/legitimate2.html /investigation/legitimate2.html
COPY investigation/mixed_techniques.html /investigation/mixed_techniques.html
COPY investigation/page_with_js.html /investigation/page_with_js.html
COPY investigation/nested_frames.html /investigation/nested_frames.html
COPY investigation/blog_post.html /investigation/blog_post.html
COPY investigation/opacity_attack_advanced.html /investigation/opacity_attack_advanced.html

RUN pip3 install --no-cache-dir --break-system-packages beautifulsoup4 lxml

WORKDIR /workspace