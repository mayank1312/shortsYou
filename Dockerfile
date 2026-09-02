FROM python:3.11-slim

# ffmpeg/ffprobe are required by app/shared/downloader.py for audio
# extraction and duration checks. Render's native Python runtime does
# not include them, so we own the image via Docker instead.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so this layer is cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Render injects PORT at runtime and routes traffic to whatever port
# the container listens on. We bind to it directly rather than relying
# on the app's own APP_PORT setting (which defaults to 8000 and is
# never read by a launch command inside the repo).
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]