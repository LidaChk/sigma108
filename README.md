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
```
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
```
*   **`frontend/`**

## Задачи (To-Do)

- [ ] **Backend:**
    - [x] Написать endpoint `/upload` для загрузки CSV (FastAPI)
    - [x] placeholder
    - [x] Реализовать endpoint `/status`
    - [x] Реализовать endpoint `/download` для выдачи обработанного CSV
    - [ ] Создать `Dockerfile` для backend
    - [ ] Добавить логику обработки ошибок (валидация файла и т.д.)
    - [ ] Логика хранения задач - сейчас processing_tasks -> Redis? Bd?
    - [ ] очитска файлов после отдачи файла или по таймауту
- [ ] **Frontend:**
    - [ ] Инициализировать React-app
    - [ ] добавить Mantine
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

**`backend/`**
```
cd ./backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```


загрузка файла
```
curl -X POST "http://localhost:8000/upload/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@"./testdata/sigma.csv""
```
Остальные запросы можно дернуть через интерфейс
`http://localhost:8000/docs`


## Ошибки (To-Do)

- [ ] **Backend:**
    - [ ] Не возращается ошибка на фронт при отсутвии колонок
- [ ] **Frontend:**
    - [ ] Отоюражение ошибок
