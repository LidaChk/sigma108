# backend/services/file_processor.py

import pandas as pd
from pathlib import Path
import numpy as np
# Используем относительный импорт: поднимаемся на уровень вверх (в backend), затем импортируем config
from config import CSV_COLUMN_CONFIG, QUESTION_NUMBER_COLUMN_NAME, SCORE_COLUMN_NAME, CSV_SEPARATOR, CSV_ENCODING

def process_csv_placeholder(input_file_path: Path, output_file_path: Path):
    """
    Placeholder-функция для обработки CSV-файла.
    Читает файл, генерирует случайные оценки в зависимости от question_number,
    и сохраняет результат в новый CSV.
    """
    try:
        df = pd.read_csv(input_file_path, sep=CSV_SEPARATOR, encoding=CSV_ENCODING)

        required_columns = set(CSV_COLUMN_CONFIG["INPUT_COLUMNS"])
        if not all(col in df.columns for col in required_columns):
             missing_cols = required_columns - set(df.columns)
             raise ValueError(f"CSV файл должен содержать колонки: {required_columns}. Отсутствуют: {missing_cols}")

        if QUESTION_NUMBER_COLUMN_NAME not in df.columns:
            raise ValueError(f"CSV файл должен содержать колонку: {QUESTION_NUMBER_COLUMN_NAME}")

        score_ranges = {
            1: (0, 1),
            2: (0, 2),
            3: (0, 1),
            4: (0, 2)
        }

        scores = []
        for _, row in df.iterrows():
            q_num = row[QUESTION_NUMBER_COLUMN_NAME]
            if q_num in score_ranges:
                min_score, max_score = score_ranges[q_num]
                score = np.random.randint(min_score, max_score + 1)
            else:
                score = 0
            scores.append(score)

        df[SCORE_COLUMN_NAME] = scores
        df.to_csv(output_file_path, index=False, encoding='utf-8', sep=';')
        print(f"Placeholder обработка завершена. Результат сохранён в {output_file_path}")

    except Exception as e:
        print(f"Ошибка в placeholder-обработке: {e}")
        if output_file_path.exists():
            output_file_path.unlink(missing_ok=True)
        raise e
