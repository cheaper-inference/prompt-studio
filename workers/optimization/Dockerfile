FROM python:3.11.13-slim
WORKDIR /worker
COPY requirements.lock .
RUN pip install --no-cache-dir --require-hashes -r requirements.lock
COPY . .
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8091
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8091", "--workers", "1", "--no-access-log"]
