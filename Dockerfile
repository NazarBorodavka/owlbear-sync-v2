FROM python:3.12-slim-bookworm

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Deliberately small next to v1's Dockerfile: the detection/decode/track
# core (tracker/vision/{geometry,codebook,decode,detect,track,pipeline}.py)
# is pure numpy with no native dependency at all, and calibration/video I/O
# only needs the opencv-python-headless WHEEL (a prebuilt binary) -- v1
# spent most of its build compiling CCTag's C++ from source (cmake, boost,
# eigen, a multi-minute build step), which this architecture has no
# equivalent of. ffmpeg is still needed as a real system binary for RTSP
# capture (the FFmpeg bundled inside the opencv wheel works fine for
# software decode; see tracker/capture.py's docstring for the RTSP tuning
# that depends on it).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY tracker/requirements.txt ./tracker/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r tracker/requirements.txt

RUN python -c "import cv2; print(f'[OK] cv2 version: {cv2.__version__}')" || \
    (echo "[ERROR] cv2 import failed" && exit 1)

COPY tracker ./tracker
COPY extension ./extension
# Shipped so an operator can generate printable marker artwork and score
# recorded footage on the deployment host itself:
#   docker exec <container> python tools/generate_markers.py --outdir /data/markers
COPY tools ./tools

# A dedicated, mountable state directory. Without this the default would
# be ./data relative to the package (/app/tracker/data), so bind-mounting
# it for persistence would shadow shipped files -- which is how
# codebook_data.json once got hidden and broke startup. Shipped code and
# user state stay in separate trees.
ENV TRACKER_DATA_DIR=/data
RUN mkdir -p /data
VOLUME ["/data"]

ARG BUILD_VERSION=dev
ARG BUILD_BRANCH=local
ARG BUILD_COMMIT=unknown
ENV BUILD_VERSION=$BUILD_VERSION
ENV BUILD_BRANCH=$BUILD_BRANCH
ENV BUILD_COMMIT=$BUILD_COMMIT

EXPOSE 5000
CMD ["python", "-m", "tracker.web.app"]
