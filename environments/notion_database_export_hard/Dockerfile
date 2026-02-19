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
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages beautifulsoup4

RUN mkdir -p /workspace/notion_export/campaigns /workspace/output

COPY data/notion_export/campaigns/Summer_Sale_Campaign_a1b2c3d4.html /workspace/notion_export/campaigns/
COPY data/notion_export/campaigns/Brand_Awareness_Initiative_e5f6g7h8.html /workspace/notion_export/campaigns/
COPY data/notion_export/campaigns/Holiday_Campaign_2024_i9j0k1l2.html /workspace/notion_export/campaigns/
COPY data/notion_export/campaigns/Social_Media_Blitz_m3n4o5p6.html /workspace/notion_export/campaigns/
COPY data/notion_export/campaigns/Q2_Product_Launch_q7r8s9t0.html /workspace/notion_export/campaigns/
COPY data/notion_export/campaigns/Email_Newsletter_Series_u1v2w3x4.html /workspace/notion_export/campaigns/
COPY data/notion_export/campaigns/Partnership_Outreach_y5z6a7b8.html /workspace/notion_export/campaigns/
COPY data/notion_export/campaigns/Influencer_Collaboration_c9d0e1f2.html /workspace/notion_export/campaigns/

WORKDIR /workspace