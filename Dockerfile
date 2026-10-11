# CPU image serving the challenge API on port 7007.
# Put detection.pt and resilience.pt in weights/ before building.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY aac/ aac/
COPY main.py classifier.py template_extractor.py run_detection_track.py run_resilience_track.py ./
COPY weights/ weights/

EXPOSE 7007
CMD ["python", "main.py"]
