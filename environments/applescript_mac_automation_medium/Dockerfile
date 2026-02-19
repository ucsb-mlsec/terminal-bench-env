FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /root/Applications/TextEditor.app/Contents && \
    mkdir -p /root/Applications/Utils/Calculator.app/Contents && \
    mkdir -p /root/Applications/DevTools/CodeEditor.app/Contents && \
    mkdir -p /root/Applications/DevTools/Terminal.app/Contents && \
    mkdir -p /tmp

COPY Applications/TextEditor.app/Contents/Info.plist /root/Applications/TextEditor.app/Contents/Info.plist
COPY Applications/Utils/Calculator.app/Contents/Info.plist /root/Applications/Utils/Calculator.app/Contents/Info.plist
COPY Applications/DevTools/CodeEditor.app/Contents/Info.plist /root/Applications/DevTools/CodeEditor.app/Contents/Info.plist
COPY Applications/DevTools/Terminal.app/Contents/Info.plist /root/Applications/DevTools/Terminal.app/Contents/Info.plist

WORKDIR /root