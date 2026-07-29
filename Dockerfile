FROM python:3.10-slim

# System libraries required by mediapipe/opencv at runtime (no GUI toolkit
# needed since opencv-python-headless is used, but these are still linked).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads outputs

ENV PYTHONUNBUFFERED=1

# Render assigns the external port dynamically via $PORT at runtime.
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 120 app:app
