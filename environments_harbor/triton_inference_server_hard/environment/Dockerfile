FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages protobuf

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/model_repository/image_classifier/1 \
    /workspace/model_repository/text_processor/1 \
    /workspace/model_repository/object_detector/1 \
    /workspace/model_repository/sentiment_analyzer \
    /workspace/model_repository/face_recognition \
    /workspace/model_repository/speech_recognizer/1 \
    /workspace/model_repository/preprocessing_pipeline/1 \
    /workspace/model_repository/translation_ensemble/1 \
    /workspace/model_repository/recommender/2 \
    /workspace/model_repository/anomaly_detector/1

COPY model_repository/image_classifier/config.pbtxt /workspace/model_repository/image_classifier/config.pbtxt
COPY model_repository/image_classifier/1/.gitkeep /workspace/model_repository/image_classifier/1/.gitkeep
COPY model_repository/text_processor/config.pbtxt /workspace/model_repository/text_processor/config.pbtxt
COPY model_repository/text_processor/1/.gitkeep /workspace/model_repository/text_processor/1/.gitkeep
COPY model_repository/object_detector/config.pbtxt /workspace/model_repository/object_detector/config.pbtxt
COPY model_repository/object_detector/1/.gitkeep /workspace/model_repository/object_detector/1/.gitkeep
COPY model_repository/sentiment_analyzer/config.pbtxt /workspace/model_repository/sentiment_analyzer/config.pbtxt
COPY model_repository/face_recognition/config.pbtxt /workspace/model_repository/face_recognition/config.pbtxt
COPY model_repository/speech_recognizer/config.pbtxt /workspace/model_repository/speech_recognizer/config.pbtxt
COPY model_repository/speech_recognizer/1/.gitkeep /workspace/model_repository/speech_recognizer/1/.gitkeep
COPY model_repository/preprocessing_pipeline/config.pbtxt /workspace/model_repository/preprocessing_pipeline/config.pbtxt
COPY model_repository/preprocessing_pipeline/1/.gitkeep /workspace/model_repository/preprocessing_pipeline/1/.gitkeep
COPY model_repository/translation_ensemble/config.pbtxt /workspace/model_repository/translation_ensemble/config.pbtxt
COPY model_repository/translation_ensemble/1/.gitkeep /workspace/model_repository/translation_ensemble/1/.gitkeep
COPY model_repository/recommender/config.pbtxt /workspace/model_repository/recommender/config.pbtxt
COPY model_repository/recommender/2/.gitkeep /workspace/model_repository/recommender/2/.gitkeep
COPY model_repository/anomaly_detector/config.pbtxt /workspace/model_repository/anomaly_detector/config.pbtxt
COPY model_repository/anomaly_detector/1/.gitkeep /workspace/model_repository/anomaly_detector/1/.gitkeep

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
