# --- Stage 1: Build Frontend ---
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install


COPY frontend/ ./

RUN npm run build


FROM python:3.12-slim

WORKDIR /app/backend

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY backend/ ./


RUN mkdir -p log
RUN mkdir -p processed
RUN mkdir -p uploads

COPY --from=frontend-builder /app/frontend/dist /app/backend/static


# Загрузка модели из приватного репозитория
RUN apt-get update && \
  apt-get install -y --no-install-recommends curl && \
  mkdir -p /app/backend/fine_tuned_rubert_base && \
  curl -fL \
  -o /app/backend/fine_tuned_rubert_base/model.safetensors \
  https://huggingface.co/LidaChk/sigma/resolve/main/model.safetensors && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
