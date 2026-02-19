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
    apt-get install -y \
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

RUN mkdir -p /var/elasticsearch_incident /home/user

COPY data/node1_elasticsearch.yml /var/elasticsearch_incident/node1_elasticsearch.yml
COPY data/node2_elasticsearch.yml /var/elasticsearch_incident/node2_elasticsearch.yml
COPY data/node3_elasticsearch.yml /var/elasticsearch_incident/node3_elasticsearch.yml
COPY data/node4_elasticsearch.yml /var/elasticsearch_incident/node4_elasticsearch.yml
COPY data/node5_elasticsearch.yml /var/elasticsearch_incident/node5_elasticsearch.yml
COPY data/node1_cluster_state.json /var/elasticsearch_incident/node1_cluster_state.json
COPY data/node3_cluster_state.json /var/elasticsearch_incident/node3_cluster_state.json
COPY data/node4_cluster_state.json /var/elasticsearch_incident/node4_cluster_state.json
COPY data/node5_cluster_state.json /var/elasticsearch_incident/node5_cluster_state.json
COPY data/node1_visibility.json /var/elasticsearch_incident/node1_visibility.json
COPY data/node2_visibility.json /var/elasticsearch_incident/node2_visibility.json
COPY data/node3_visibility.json /var/elasticsearch_incident/node3_visibility.json
COPY data/node4_visibility.json /var/elasticsearch_incident/node4_visibility.json
COPY data/node5_visibility.json /var/elasticsearch_incident/node5_visibility.json
COPY data/incident_log.txt /var/elasticsearch_incident/incident_log.txt

WORKDIR /home/user