# Sigma108 App

**Цель:** Создать веб-приложение для автоматической оценки устных ответов на экзамене по русскому языку для иностранных граждан.

**Деплой:** [https://lidachk.github.io/sigma108/](https://lidachk.github.io/sigma108/) 

**Видео:** [https://cloud.mail.ru/public/xcKe/UBnpmvgwz](https://cloud.mail.ru/public/xcKe/UBnpmvgwz)

**Инструкция:** [instruction.md](instruction.md)

## Стек

- **Backend:** Python, FastAPI
- **Frontend:** React, Gravity UI
- **Контейнеризация:** Docker
- **CI/CD:** GitHub Actions
- **Развёртывание:** Yandex Cloud
- **ML:** Python, transformers, torch, pandas
- **SLM:** ruBert-base (open source - [ruBert-base на Hugging Face](https://huggingface.co/ai-forever/ruBert-base) - trained by torch, peft, transformers, pandas)

## Структура проекта

- **`backend/`**

```
├── main.py                      # FastAPI, подключение маршрутов
├── config.py                    #
├── Dockerfile                   #
├── fine_tuned_rubert_base/      # Модель
│   ├── config.json              #
│   ├── model.safetensors        #
│   ├── special_tokens_map.json  #
│   ├── tokenizer.json           #
│   ├── tokenizer_config.json    #
│   ├── training_args.bin        #
│   └── vocab.txt                #
├── services/                    # Логика приложения
│   ├── __init__.py              #
│   ├── file_processor.py        # process_csv_placeholder
│   └── task_manager.py          # processing_tasks, стату
├── api/                         # роутеры FastAPI
│   ├── __init__.py              #
│   └── routes.py                # @app.post("/upload"), @app.get("/status"), @app.get("/download")
├── logger/                      #
│   └── log_config.py            # Логирование
└── requirements.txt             # Зависимости для backend (fastapi, uvicorn, pandas, numpy)
```

- **`frontend/`**

## Задачи (To-Do)

- [ ] **Backend:**
  - [x] Написать endpoint `/upload` для загрузки CSV (FastAPI)
  - [x] placeholder
  - [x] Реализовать endpoint `/status`
  - [x] Реализовать endpoint `/download` для выдачи обработанного CSV
  - [x] Создать `Dockerfile` для backend
  - [x] Добавить логику обработки ошибок (валидация файла и т.д.)
  - [ ] очитска файлов после отдачи файла или по таймауту
  - [x] отправлять актуальный (%) статус обрабоки файла
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
- [x] **ML-логика:**
  - [x] выбрать наиболее подходящую модельку
  - [x] перевести картинки в текст
  - [x] подготовить данные для дообучения
  - [x] код для дообучения
  - [x] код для приема .csv и предсказания ответов

## Запуск

**`docker`**

```bash
docker-compose up --build
```

**`frontend/`**

```bash
cd ./frontend
npm run dev
```

**`backend/`**

для bash VScode с Conda

```bash
eval "$(conda shell.bash hook)"
conda activate auto-eval
```

```bash
cd ./backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
