FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for OpenCV and camera streaming
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV MQTT_BROKER=mar-mosquitto
ENV MQTT_PORT=1883

EXPOSE 8080

CMD ["python", "main.py", "--headless", "--depth"]
