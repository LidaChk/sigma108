import os
from pathlib import Path

# Конфигурация названий колонок для CSV-файлов

CSV_COLUMN_CONFIG = {
    # Входной файл (без оценки экзаменатора)
    "INPUT_COLUMNS": [
        "Id экзамена",            # int: Уникальный идентификатор экзамена
        "Id вопроса",             # int: Уникальный идентификатор вопроса
        "№ вопроса",              # int: Номер вопроса в экзамене (1-4)
        "Текст вопроса",          # string: Текст вопроса (может содержать HTML)
        "Картинка из вопроса",    # string (URL, optional): Ссылка на картинку (если есть)
        "Оценка экзаменатора",
        "Транскрибация ответа",   # string: Текст транскрибации ответа
        "Ссылка на оригинальный файл запис" # string (URL): Ссылка на аудиофайл
    ],
    # Выходной файл (с добавленной оценкой инструмента)
    "OUTPUT_COLUMNS": [
        "Id экзамена",
        "Id вопроса",
        "№ вопроса",
        "Текст вопроса",
        "Картинка из вопроса",
        "Оценка экзаменатора", # int: Оценка, добавляемая инструментом (placeholder или ML)
        "Транскрибация ответа",
        "Ссылка на оригинальный файл запис",
        "Уверенность предсказания"
    ],
    # Колонка, которую мы генерируем
    "SCORE_COLUMN": "Оценка экзаменатора",
    "QUESTION_NUMBER_COLUMN": "№ вопроса"
}

INPUT_FILE_REQUIRED_COLUMNS = set(CSV_COLUMN_CONFIG["INPUT_COLUMNS"])
OUTPUT_FILE_EXPECTED_COLUMNS = set(CSV_COLUMN_CONFIG["OUTPUT_COLUMNS"])
SCORE_COLUMN_NAME = CSV_COLUMN_CONFIG["SCORE_COLUMN"]
QUESTION_NUMBER_COLUMN_NAME = CSV_COLUMN_CONFIG["QUESTION_NUMBER_COLUMN"]

# --- Конфигурация файлов ---
BACKEND_PATH = Path(__file__).resolve().parent
# Папки для файлов
UPLOADS_DIR = BACKEND_PATH / "uploads"
PROCESSED_DIR = BACKEND_PATH / "processed"

# Создаём папки, если их нет
UPLOADS_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# --- Конфигурация CSV ---
CSV_SEPARATOR = ';'
CSV_ENCODING = 'utf-8'

# --- Конфигурация статусов задач ---
TASK_STATUS = {
    "CREATED": "created",
    "PROCESSING": "processing",
    "COMPLETED": "completed",
    "FAILED": "failed"
}
