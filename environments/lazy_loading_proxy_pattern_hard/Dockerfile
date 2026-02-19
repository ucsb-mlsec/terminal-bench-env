FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/plugin_system/plugins

COPY plugin_system/__init__.py /workspace/plugin_system/__init__.py
COPY plugin_system/plugins/__init__.py /workspace/plugin_system/plugins/__init__.py
COPY plugin_system/plugins/database_plugin.py /workspace/plugin_system/plugins/database_plugin.py
COPY plugin_system/plugins/api_plugin.py /workspace/plugin_system/plugins/api_plugin.py
COPY plugin_system/plugins/cache_plugin.py /workspace/plugin_system/plugins/cache_plugin.py
COPY plugin_system/plugins/logging_plugin.py /workspace/plugin_system/plugins/logging_plugin.py
COPY plugin_system/plugins/analytics_plugin.py /workspace/plugin_system/plugins/analytics_plugin.py
COPY plugin_system/plugins/search_plugin.py /workspace/plugin_system/plugins/search_plugin.py
COPY plugin_system/plugins/notification_plugin.py /workspace/plugin_system/plugins/notification_plugin.py
COPY plugin_system/plugins/auth_plugin.py /workspace/plugin_system/plugins/auth_plugin.py
COPY plugin_system/test_lazy_loading.py /workspace/plugin_system/test_lazy_loading.py

WORKDIR /workspace