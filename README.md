# Sigma108 App

Цель: Создать веб-приложение для автоматической оценки устных ответов на экзамене по русскому языку для иностранных граждан.

## Стек

*   **Backend:** Python, FastAPI
*   **Frontend:** React, Gravity UI
*   **Контейнеризация:** Docker
*   **CI/CD:** GitHub Actions
*   **Развёртывание:** Yandex Cloud
*   **ML:** Python, transformers, torch, pandas
*   **SLM:** ruBert-base (open source - https://huggingface.co/ai-forever/ruBert-base - trained by torch, peft, transformers, pandas)

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
    - [x] Создать `Dockerfile` для backend
    - [ ] Добавить логику обработки ошибок (валидация файла и т.д.)
    - [ ] очитска файлов после отдачи файла или по таймауту
    - [ ] отправлять актуальный (%) статус обрабоки файла
    - [ ] толкьо одна задача одновременно выполняется
    - [ ] номер в очереди
- [ ] **Frontend:**
    - [x] Инициализировать React-app
    - [x] добавить Mantine
    - [x] компонент для загрузки файла
    - [x] компонент для отображения статуса обработки
    - [x] компонент для скачивания результата
    - [x] Настроить вызов backend API из frontend
    - [ ] ссылки на github
    - [ ] отображать актуальный статус обработки файла
    - [ ] отображать время затраченное на обработку
- [x] **Docker:**
    - [x] Написать `docker-compose.yml` для локального запуска backend + frontend
- [ ] **CI/CD & Deploy:**
    - [ ] создать ветку Deploy
- [ ] **ML-логика:**


## Запуск

**`docker`**
docker-compose up --build


**`frontend/`**
```
cd ./frontend
npm run dev
```


**`backend/`**

для bash VScode с Conda
```
eval "$(conda shell.bash hook)"
conda activate auto-eval
```

```
cd ./backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Ошибки (To-Do)

- [ ] **Backend:**
    - [ ] Не возращается ошибка на фронт при отсутвии колонок
- [ ] **Frontend:**
    - [ ] Отображение ошибок
