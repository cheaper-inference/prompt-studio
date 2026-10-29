# Existing Railway service: web and durable worker are separate supervised processes.
# They share the service's persistent SQLite/MLflow volume.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build tools for the few source-only wheels in the ML stack. Removed in the
# same layer to keep the image small.
WORKDIR /app

COPY requirements.txt ./
RUN apt-get update \
    && apt-get install -y procps \
    && apt-get install -y vim \
    && apt-get install -y sqlite3 \
    && apt-get install -y --no-install-recommends build-essential \
    && python -m pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY llm_endpoint.py prompt_optimizer.py run_service.py sentry_init.py ./
COPY ps_utils ./ps_utils
COPY webapp_django ./webapp_django
COPY test ./test

# /app/data is the mount point for the Railway volume. Both the app DB and the
# MLflow store live here so they persist across redeploys.
RUN mkdir -p /app/data

ENV DJANGO_SETTINGS_MODULE=optimizer_site.settings \
    PYTHONPATH=/app \
    DB_PATH=/app/data/db.sqlite3 \
    MLFLOW_TRACKING_URI=sqlite:////app/data/mlflow.db \
    MINIME_DATA_DIR=/app/data/minime \
    PORT=8000

WORKDIR /app/webapp_django

EXPOSE 8000

# Migrate before starting either process; the supervisor restarts the service if
# either process exits. No second Railway project or shared-volume service needed.
CMD ["python", "/app/run_service.py"]
