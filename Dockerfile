# The image that runs the API.
#   docker build -t intrusion-api .
#   docker run --rm -p 8000:8000 intrusion-api
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dependencies before the code: editing api.py then rebuilds in seconds.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Only what the service needs at runtime.
COPY src/api.py src/explain.py src/data.py src/
COPY models/ models/

# A service should not run with administrator rights.
RUN useradd --create-home --uid 1000 service
USER service

EXPOSE 8000

# 0.0.0.0 means "accept connections from outside this container".
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
