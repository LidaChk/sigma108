import pandas as pd
import numpy as np
import torch
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import CSV_COLUMN_CONFIG, QUESTION_NUMBER_COLUMN_NAME, SCORE_COLUMN_NAME, CSV_ENCODING
from logger.log_config import logger


def clean_text(text):
    """Очистка текста от HTML-тегов и лишних символов"""
    if pd.isna(text):
        return ""

    text = str(text)
    # Удаляем HTML-теги
    text = re.sub(r'<[^>]+>', '', text)
    # Удаляем специальные символы, но сохраняем кириллицу, латиницу и пунктуацию
    text = re.sub(r'[^\w\sа-яА-ЯёЁ.,!?;:()\-]', '', text)
    # Заменяем множественные пробелы и переносы на один пробел
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

class ModelPredictor:
        def __init__(self, model_path: str = None):
        """Инициализация модели для предсказания оценок"""
        try:
            # 🔧 АВТОМАТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ ПУТИ К МОДЕЛИ
            if model_path is None:
                # Определяем базовую директорию (текущий файл: backend/services/file_processor.py)
                current_file = Path(__file__)
                base_dir = current_file.parent.parent  # backend/

                # Основной путь к модели: backend/fine_tuned_rubert_base
                model_path = str(base_dir / "fine_tuned_rubert_base")

                logger.info(f"🔍 Поиск модели по пути: {model_path}")

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"🎯 Используемое устройство: {self.device}")
            logger.info(f"📁 Загрузка модели из: {model_path}")

            # 🔧 ПРОВЕРКА СУЩЕСТВОВАНИЯ МОДЕЛИ
            model_dir = Path(model_path)
            if not model_dir.exists():
                error_msg = f"❌ Директория с моделью не найдена: {model_path}"
                logger.error(error_msg)

                # Покажем что есть в директории
                if model_dir.parent.exists():
                    available = [f.name for f in model_dir.parent.iterdir() if f.is_dir()]
                    logger.error(f"📂 Доступные директории в {model_dir.parent}: {available}")

                raise FileNotFoundError(error_msg)

            # 🔧 ЗАГРУЗКА С ЛОКАЛЬНЫМИ ФАЙЛАМИ
            logger.info("🔄 Загрузка токенизатора...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir),
                local_files_only=True
            )

            logger.info("🔄 Загрузка модели...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(model_dir),
                local_files_only=True
            )

            self.model.to(self.device)
            self.model.eval()

            # Диапазоны баллов для каждого вопроса
            self.question_scores = {
                1: (0, 1),  # Вопрос 1: от 0 до 1 балла
                2: (0, 2),  # Вопрос 2: от 0 до 2 баллов
                3: (0, 1),  # Вопрос 3: от 0 до 1 балла
                4: (0, 2)  # Вопрос 4: от 0 до 2 баллов
            }

            logger.info("✅ Модель успешно загружена и готова к предсказаниям!")

        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке модели: {e}")
            raise

    def create_training_text(self, question_text: str, answer_transcription: str, question_num) -> str:
        """Создание текста для модели в формате: Текст вопроса + Транскрибация ответа"""
        question_clean = self.preprocess_text(question_text)
        answer_clean = self.preprocess_text(answer_transcription)

        if question_num == 4:
            training_text = answer_clean
        else:
            training_text = f"Вопрос: {question_clean}\n\nОтвет: {answer_clean}"
        return training_text

    def preprocess_text(self, text: str) -> str:
        """Предобработка текста с очисткой от HTML-тегов"""
        if pd.isna(text) or text is None:
            return ""

        # Используем функцию clean_text для очистки
        text = clean_text(text)

        # Дополнительная обработка для модели
        text = str(text).strip()
        text = ' '.join(text.split())
        return text

    def predict_single_text(self, text: str) -> tuple[int, str]:
        """Предсказание для одного текста с возвратом всех вероятностей"""
        try:
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )

            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=1).item()

                # Получаем все вероятности для каждого класса
                probs = predictions[0].cpu().numpy()
                # Форматируем в строку типа "0=0.4771, 1=0.5090, 2=0.0138"
                confidence_str = ", ".join([f"{i}={probs[i]:.4f}" for i in range(len(probs))])

            return predicted_class, confidence_str

        except Exception as e:
            logger.error(f"Ошибка при предсказании: {e}")
            return 0, "0=0.0000, 1=0.0000, 2=0.0000"

    def map_class_to_score(self, predicted_class: int, question_number: int) -> int:
        """Преобразование предсказанного класса в баллы"""
        if question_number not in self.question_scores:
            logger.warning(f"Неизвестный номер вопроса: {question_number}, установлен 0")
            return 0

        min_score, max_score = self.question_scores[question_number]

        if max_score == 1:  # Вопросы 1 и 3 (0-1 балл)
            return 0 if predicted_class == 0 else 1
        else:  # Вопросы 2 и 4 (0-2 балла)
            return min(predicted_class, max_score)


def process_csv_with_model(input_file_path: Path, output_file_path: Path, model_path: str = None):
    """
    Основная функция для обработки CSV-файла с помощью модели.
    Читает файл, предсказывает оценки и сохраняет результат.
    """
    logger.info(f"Начало обработки файла: {input_file_path}")

    try:
        # Загрузка данных
        try:
            df = pd.read_csv(input_file_path, sep=';', encoding=CSV_ENCODING)
            logger.info("Файл прочитан с разделителем ';'")
        except pd.errors.ParserError:
            df = pd.read_csv(input_file_path, sep=',', encoding=CSV_ENCODING)
            logger.info("Файл прочитан с разделителем ','")

        # Проверка обязательных колонок
        required_columns = set(CSV_COLUMN_CONFIG["INPUT_COLUMNS"])
        if not all(col in df.columns for col in required_columns):
            missing_cols = required_columns - set(df.columns)
            error_msg = f"CSV файл должен содержать колонки: {required_columns}. Отсутствуют: {missing_cols}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if QUESTION_NUMBER_COLUMN_NAME not in df.columns:
            error_msg = f"CSV файл должен содержать колонку: {QUESTION_NUMBER_COLUMN_NAME}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Инициализация модели
        logger.info("Загрузка модели...")
        predictor = ModelPredictor(model_path)

        # Предсказание оценок
        logger.info("Начало предсказания оценок...")
        scores = []
        confidences = []
        processed_count = 0

        for idx, row in df.iterrows():
            question_num = row[QUESTION_NUMBER_COLUMN_NAME]
            question_text = row.get('Текст вопроса', '')
            answer_text = row.get('Транскрибация ответа', '')

            # Создание текста для модели
            training_text = predictor.create_training_text(question_text, answer_text, question_num)

            if not training_text or training_text == "Вопрос: \n\nОтвет: ":
                logger.warning(f"Пустые данные для строки {idx}, вопрос {question_num}")
                score = 0
                confidence = 0.0
            else:
                # Предсказание
                predicted_class, confidence = predictor.predict_single_text(training_text)
                score = predictor.map_class_to_score(predicted_class, question_num)
                processed_count += 1

            scores.append(score)
            confidences.append(confidence)


            # Логирование прогресса
            if (idx + 1) % 100 == 0 or (idx + 1) == len(df):
                progress_percent = int(((idx + 1) / len(df)) * 100)  # TODO процент обработанных на текущий момент
                logger.info(f"Обработано {idx + 1}/{len(df)} записей ({progress_percent}%)")


        # Заполнение колонки с оценками
        df[SCORE_COLUMN_NAME] = scores
        df['Уверенность предсказания'] = confidences

        # Сохранение результата
        df.to_csv(output_file_path, index=False, encoding='utf-8', sep=';')
        logger.info(
            f"✅ Обработка завершена. Обработано {processed_count} записей. Результат сохранён в {output_file_path}")

        return True

    except Exception as e:
        logger.error(f"❌ Ошибка при обработке файла: {e}")
        if output_file_path.exists():
            output_file_path.unlink(missing_ok=True)
            logger.info("Удален частично созданный выходной файл")
        raise e


def process_csv_placeholder(input_file_path: Path, output_file_path: Path):
    """
    Резервная функция-заглушка для обработки CSV-файла.
    Используется если модель недоступна.
    """
    logger.warning("Используется placeholder-режим (случайные оценки)")

    try:
        try:
            df = pd.read_csv(input_file_path, sep=';', encoding=CSV_ENCODING)
        except pd.errors.ParserError:
            df = pd.read_csv(input_file_path, sep=',', encoding=CSV_ENCODING)

        required_columns = set(CSV_COLUMN_CONFIG["INPUT_COLUMNS"])
        if not all(col in df.columns for col in required_columns):
            missing_cols = required_columns - set(df.columns)
            error_msg = f"CSV файл должен содержать колонки: {required_columns}. Отсутствуют: {missing_cols}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if QUESTION_NUMBER_COLUMN_NAME not in df.columns:
            error_msg = f"CSV файл должен содержать колонку: {QUESTION_NUMBER_COLUMN_NAME}"
            logger.error(error_msg)
            raise ValueError(error_msg)

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
        logger.info(f"Placeholder обработка завершена. Результат сохранён в {output_file_path}")

    except Exception as e:
        logger.error(f"Ошибка в placeholder-обработке: {e}")
        if output_file_path.exists():
            output_file_path.unlink(missing_ok=True)
        raise e


# Основная функция для вызова из бэкенда
def process_exam_csv(input_file_path: Path, output_file_path: Path, use_model: bool = True):
    """
    Основная функция для обработки экзаменационных CSV-файлов.

    Args:
        input_file_path: Путь к входному CSV-файлу
        output_file_path: Путь для сохранения результата
        use_model: Использовать ли модель для предсказания (True) или заглушку (False)
    """
    try:
        if use_model:
            return process_csv_with_model(input_file_path, output_file_path)
        else:
            return process_csv_placeholder(input_file_path, output_file_path)
    except Exception as e:
        logger.error(f"Критическая ошибка при обработке CSV: {e}")
        raise
