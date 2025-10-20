# Sigma108 App

Цель: Создать веб-приложение для автоматической оценки устных ответов на экзамене по русскому языку для иностранных граждан.

## Стек

*   **Backend:** Python, FastAPI
*   **Frontend:** React, Gravity UI
*   **Контейнеризация:** Docker
*   **CI/CD:** GitHub Actions
*   **Развёртывание:** Yandex Cloud
*   **ML:**

## Структура проекта

*   **`backend/`**
backend/
├── main.py               # FastAPI, подключение маршрутов
├── config.py             #
├── models/               #
│   └── ...
├── services/             # Логика приложения
│   ├── __init__.py       #
│   ├── file_processor.py # process_csv_placeholder
│   └── task_manager.py   # processing_tasks, стату
├── api/                  # роутеры FastAPI
│   ├── __init__.py       #
│   └── routes.py         # @app.post("/upload"), @app.get("/status"), @app.get("/download")
└── requirements.txt      # Зависимости для backend (fastapi, uvicorn, pandas, numpy)
*   **`frontend/`**

## Задачи (To-Do)

- [ ] **Backend:**
    - [ ] Написать endpoint `/upload` для загрузки CSV (FastAPI)
    - [ ] placeholder
    - [ ] Реализовать endpoint `/download` для выдачи обработанного CSV
    - [ ] Создать `Dockerfile` для backend
    - [ ] Добавить логику обработки ошибок (валидация файла и т.д.)
    - [ ] Логика хранения задач - сейчас processing_tasks -> Redis? Bd?
- [ ] **Frontend:**
    - [ ] Инициализировать React-app
    - [ ] добавить Gravity UI
    - [ ] компонент для загрузки файла
    - [ ] компонент для отображения статуса обработки
    - [ ] компонент для скачивания результата
    - [ ] Настроить вызов backend API из frontend
    - [ ] Создать `Dockerfile` для frontend
- [ ] **Docker:**
    - [ ] Написать `docker-compose.yml` для локального запуска backend + frontend
- [ ] **CI/CD & Deploy:**
    - [ ] Настроить секреты в GitHub (YC CLI creds, Service Account key)
    - [ ] Заполнить `.github/workflows/deploy.yml`
- [ ] **ML-логика:**


## Запуск
