# --- Stage 1: Build Frontend ---
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install


COPY frontend/ ./

RUN npm run build


FROM python:3.12-slim

WORKDIR /app


# УСТАНАВЛИВАЕМ ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ ДЛЯ TRANSFORMERS
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# КОПИРУЕМ ВЕСЬ BACKEND
COPY backend/ ./backend/

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./

RUN mkdir -p log processed uploads

# Copy the pulled model from temp clone

COPY --from=frontend-builder /app/frontend/dist /app/backend/static

WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
