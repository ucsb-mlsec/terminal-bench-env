FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install steganography and forensic tools
RUN apt-get update && \
    apt-get install -y \
    steghide \
    stegseek \
    file \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y steghide && \
    rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /home/employee /tmp

# Copy all generated files to employee home directory
COPY home/employee/vacation_photo1.jpg /home/employee/vacation_photo1.jpg
COPY home/employee/vacation_photo2.jpg /home/employee/vacation_photo2.jpg
COPY home/employee/family_gathering.png /home/employee/family_gathering.png
COPY home/employee/office_party.jpg /home/employee/office_party.jpg
COPY home/employee/weekend_trip.jpg /home/employee/weekend_trip.jpg
COPY home/employee/birthday_celebration.png /home/employee/birthday_celebration.png
COPY home/employee/document1.pdf /home/employee/document1.pdf
COPY home/employee/notes.txt /home/employee/notes.txt

WORKDIR /home/employee