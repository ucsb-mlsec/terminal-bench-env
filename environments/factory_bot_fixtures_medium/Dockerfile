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
    apt-get install -y ruby ruby-dev build-essential && \
    gem install factory_bot && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/challenge /workspace/models /workspace/factories

COPY models/player.rb /workspace/models/player.rb
COPY models/game_session.rb /workspace/models/game_session.rb
COPY models/score.rb /workspace/models/score.rb
COPY factories/players.rb /workspace/factories/players.rb
COPY factories/game_sessions.rb /workspace/factories/game_sessions.rb
COPY factories/scores.rb /workspace/factories/scores.rb
COPY test_factories.rb /workspace/test_factories.rb

WORKDIR /workspace