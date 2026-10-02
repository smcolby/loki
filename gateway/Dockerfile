# Model-routing gateway in front of llama-server and Strata
FROM docker.io/library/python:3.12-slim

WORKDIR /opt/gateway
COPY requirements.txt .
RUN pip install --no-cache-dir --quiet -r requirements.txt
COPY gateway.py .

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=4)"

ENTRYPOINT ["python", "/opt/gateway/gateway.py"]
