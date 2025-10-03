# Dockerfile — EGX Intraday Assistant (analysis + webhook in one container)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only requirements first (layer caching)
COPY src/egx_intraday_enhanced/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the whole repo
COPY . /app

# Start script runs the assistant loop + uvicorn webhook
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8000
CMD ["/app/start.sh"]
