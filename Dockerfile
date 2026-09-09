# Two-stage build. The wheel layer is cached on requirements alone, so editing
# source does not reinstall numpy on every rebuild.
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt requirements-service.txt ./
RUN pip install --no-cache-dir --prefix=/install \
        -r requirements.txt -r requirements-service.txt


FROM python:3.12-slim

# Run as a non-root user. This service will hold a payment routing decision and
# an API key; there is no reason for it to be root in its own container.
RUN useradd --create-home --uid 10001 router
WORKDIR /app

COPY --from=builder /install /usr/local
COPY --chown=router:router src/ ./src/
COPY --chown=router:router service/ ./service/
COPY --chown=router:router bench_latency.py pytest.ini ./

USER router
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# /health reports which optional backends actually connected, so an unhealthy
# container is distinguishable from one merely running without Redis.
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=2).status==200 else 1)"

CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
