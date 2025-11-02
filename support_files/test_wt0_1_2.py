# compare_predictions.py
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


class PredictionComparator:
    def __init__(self, model_path: str = "./fine_tuned_rubert_base"):
        """Инициализация модели для сравнения предсказаний"""

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"🎯 Используемое устройство: {self.device}")

        # Загрузка модели и токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # 🎯 ЧЕТКИЕ ДИАПАЗОНЫ БАЛЛОВ ДЛЯ КАЖДОГО ВОПРОСА
        self.question_scores = {
            1: (0, 1),  # Вопрос 1: от 0 до 1 балла
            2: (0, 2),  # Вопрос 2: от 0 до 2 баллов
            3: (0, 1),  # Вопрос 3: от 0 до 1 балла
            4: (0, 2)  # Вопрос 4: от 0 до 2 баллов
        }

        print("✅ Модель загружена и готова к предсказаниям!")

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
        """Предобработка текста"""
        if pd.isna(text) or text is None:
            return ""
        text = str(text).strip()
        text = ' '.join(text.split())
        return text

    def predict_single_text(self, text: str) -> tuple[int, float]:
        """Предсказание для одного текста"""
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
                confidence = torch.max(predictions).item()

            return predicted_class, confidence

        except Exception as e:
            print(f"❌ Ошибка при предсказании: {e}")
            return 0, 0.0

    def map_class_to_score(self, predicted_class: int, question_number: int) -> int:
        """Преобразование предсказанного класса в баллы согласно диапазону вопроса"""
        min_score, max_score = self.question_scores[question_number]

        if max_score == 1:  # Вопросы 1 и 3 (0-1 балл)
            return 0 if predicted_class == 0 else 1
        else:  # Вопросы 2 и 4 (0-2 балла)
            return min(predicted_class, max_score)

    def predict_test_set(self, test_csv_path: str = "clear_test.csv") -> pd.DataFrame:
        """Предсказание оценок для тестового набора"""
        print("🔮 ПРЕДСКАЗАНИЕ ОЦЕНОК ДЛЯ ТЕСТОВОГО НАБОРА...")
        print("=" * 50)

        # Загрузка тестовых данных
        test_df = pd.read_csv(test_csv_path, sep=',')
        print(f"📁 Загружено {len(test_df)} тестовых записей")

        # Проверка структуры
        print(f"\n🔍 СТРУКТУРА ТЕСТОВЫХ ДАННЫХ:")
        for col in test_df.columns:
            non_null = test_df[col].notna().sum()
            null_percentage = (test_df[col].isna().sum() / len(test_df)) * 100
            print(f"   {col}: {non_null} непустых ({null_percentage:.1f}% пустых)")

        # Результаты предсказаний
        predictions = []

        print(f"\n🎯 НАЧАЛО ПРЕДСКАЗАНИЙ...")
        print("-" * 40)

        for idx, row in test_df.iterrows():
            question_num = row['№ вопроса']
            question_text = row['Текст вопроса']
            answer_text = row['Транскрибация ответа']
            exam_id = row['Id экзамена']
            question_id = row['Id вопроса']

            # Создаем объединенный текст
            training_text = self.create_training_text(question_text, answer_text, question_num)

            if not training_text or training_text == "Вопрос: \n\nОтвет: ":
                print(f"⚠️ Пустые данные для экзамена {exam_id}, вопрос {question_num}")
                predictions.append({
                    'Id экзамена': exam_id,
                    'Id вопроса': question_id,
                    '№ вопроса': question_num,
                    'predicted_class': 0,
                    'predicted_score': 0,
                    'confidence': 0.0,
                    'error': 'empty_data'
                })
                continue

            # Предсказание
            predicted_class, confidence = self.predict_single_text(training_text)
            predicted_score = self.map_class_to_score(predicted_class, question_num)

            predictions.append({
                'Id экзамена': exam_id,
                'Id вопроса': question_id,
                '№ вопроса': question_num,
                'predicted_class': predicted_class,
                'predicted_score': predicted_score,
                'confidence': confidence,
                'max_score': self.question_scores[question_num][1]
            })

            if (idx + 1) % 100 == 0:
                print(f"✅ Обработано {idx + 1}/{len(test_df)} записей...")

        predictions_df = pd.DataFrame(predictions)
        print(f"📊 Предсказания завершены для {len(predictions_df)} записей")

        return predictions_df

    def load_true_marks(self, marks_csv_path: str = "clear_test_with_marks.csv") -> pd.DataFrame:
        """Загрузка правильных оценок"""
        print(f"\n📖 ЗАГРУЗКА ПРАВИЛЬНЫХ ОЦЕНОК...")
        marks_df = pd.read_csv(marks_csv_path, sep=',')
        print(f"📁 Загружено {len(marks_df)} записей с оценками")

        # Проверяем наличие столбца с оценками
        if 'Оценка экзаменатора' not in marks_df.columns:
            print("❌ В файле с оценками нет столбца 'Оценка экзаменатора'")
            return pd.DataFrame()

        # Проверяем, есть ли оценки
        non_null_marks = marks_df['Оценка экзаменатора'].notna().sum()
        print(f"📝 Найдено {non_null_marks} непустых оценок")

        return marks_df[['Id экзамена', 'Id вопроса', '№ вопроса', 'Оценка экзаменатора']]

    def compare_predictions(self, predictions_df: pd.DataFrame, true_marks_df: pd.DataFrame):
        """Сравнение предсказаний с правильными оценками"""
        print(f"\n🔍 СРАВНЕНИЕ ПРЕДСКАЗАНИЙ С ПРАВИЛЬНЫМИ ОЦЕНКАМИ...")
        print("=" * 60)

        # Объединяем предсказания с правильными оценками
        comparison_df = pd.merge(
            predictions_df,
            true_marks_df,
            on=['Id экзамена', 'Id вопроса', '№ вопроса'],
            how='inner'
        )

        print(f"📊 Успешно сопоставлено {len(comparison_df)} записей")

        if len(comparison_df) == 0:
            print("❌ Не удалось сопоставить записи. Проверьте идентификаторы.")
            return

        # Убедимся, что оценки числовые
        comparison_df['Оценка экзаменатора'] = pd.to_numeric(comparison_df['Оценка экзаменатора'], errors='coerce')
        comparison_df = comparison_df.dropna(subset=['Оценка экзаменатора'])

        # Добавляем флаг совпадения
        comparison_df['match'] = comparison_df['predicted_score'] == comparison_df['Оценка экзаменатора']

        # Общая точность
        overall_accuracy = comparison_df['match'].mean() * 100
        print(f"\n🎯 ОБЩАЯ ТОЧНОСТЬ: {overall_accuracy:.2f}%")

        # Точность по вопросам
        print(f"\n📊 ТОЧНОСТЬ ПО ВОПРОСАМ:")
        for question_num in range(1, 5):
            question_data = comparison_df[comparison_df['№ вопроса'] == question_num]
            if len(question_data) > 0:
                accuracy = question_data['match'].mean() * 100
                max_score = self.question_scores[question_num][1]
                print(f"   Вопрос {question_num} (0-{max_score}): {accuracy:.2f}% ({len(question_data)} записей)")

        # Статистика по уверенности
        high_confidence = comparison_df[comparison_df['confidence'] > 0.7]
        if len(high_confidence) > 0:
            high_conf_accuracy = high_confidence['match'].mean() * 100
            print(f"📈 Точность при уверенности >0.7: {high_conf_accuracy:.2f}% ({len(high_confidence)} записей)")

        return comparison_df

    def generate_detailed_report(self, comparison_df: pd.DataFrame):
        """Генерация детального отчета о сравнении"""
        print(f"\n📈 ДЕТАЛЬНЫЙ ОТЧЕТ О СРАВНЕНИИ")
        print("=" * 60)

        # Матрицы ошибок для каждого вопроса
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Матрицы ошибок по вопросам', fontsize=16, fontweight='bold')

        for i, question_num in enumerate(range(1, 5)):
            ax = axes[(i) // 2, (i) % 2]
            question_data = comparison_df[comparison_df['№ вопроса'] == question_num]

            if len(question_data) > 0:
                y_true = question_data['Оценка экзаменатора']
                y_pred = question_data['predicted_score']

                # Матрица ошибок
                cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true) | set(y_pred)))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=sorted(set(y_true) | set(y_pred)),
                            yticklabels=sorted(set(y_true) | set(y_pred)))

                ax.set_title(f'Вопрос {question_num} (0-{self.question_scores[question_num][1]})')
                ax.set_xlabel('Предсказанные оценки')
                ax.set_ylabel('Правильные оценки')

        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        print(f"💾 Матрицы ошибок сохранены в 'confusion_matrices.png'")

        # Classification report для каждого вопроса
        print(f"\n📋 КЛАССИФИКАЦИОННЫЕ ОТЧЕТЫ:")
        for question_num in range(1, 5):
            question_data = comparison_df[comparison_df['№ вопроса'] == question_num]
            if len(question_data) > 0:
                y_true = question_data['Оценка экзаменатора']
                y_pred = question_data['predicted_score']

                print(f"\n❓ Вопрос {question_num}:")
                print(classification_report(y_true, y_pred, digits=3))

        # Анализ распределения оценок
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        comparison_df['Оценка экзаменатора'].value_counts().sort_index().plot(kind='bar', color='lightblue')
        plt.title('Распределение правильных оценок')
        plt.xlabel('Оценка')
        plt.ylabel('Количество')

        plt.subplot(2, 2, 2)
        comparison_df['predicted_score'].value_counts().sort_index().plot(kind='bar', color='lightgreen')
        plt.title('Распределение предсказанных оценок')
        plt.xlabel('Оценка')
        plt.ylabel('Количество')

        plt.subplot(2, 2, 3)
        comparison_df['confidence'].hist(bins=30, color='lightcoral', alpha=0.7)
        plt.title('Распределение уверенности модели')
        plt.xlabel('Уверенность')
        plt.ylabel('Количество')

        plt.subplot(2, 2, 4)
        match_by_confidence = comparison_df.groupby(pd.cut(comparison_df['confidence'], bins=10))['match'].mean()
        match_by_confidence.plot(kind='bar', color='gold')
        plt.title('Точность по уровням уверенности')
        plt.xlabel('Уровень уверенности')
        plt.ylabel('Точность')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
        print(f"💾 Анализ предсказаний сохранен в 'prediction_analysis.png'")

        # Детальная статистика
        print(f"\n📊 ДЕТАЛЬНАЯ СТАТИСТИКА:")
        print(f"   Средняя уверенность модели: {comparison_df['confidence'].mean():.3f}")
        print(
            f"   Средняя абсолютная ошибка: {abs(comparison_df['predicted_score'] - comparison_df['Оценка экзаменатора']).mean():.3f}")

        # F1-score по вопросам
        print(f"\n🎯 F1-МЕТРИКИ ПО ВОПРОСАМ:")
        for question_num in range(1, 5):
            question_data = comparison_df[comparison_df['№ вопроса'] == question_num]
            if len(question_data) > 0:
                y_true = question_data['Оценка экзаменатора']
                y_pred = question_data['predicted_score']
                f1 = f1_score(y_true, y_pred, average='weighted')
                print(f"   Вопрос {question_num}: F1 = {f1:.3f}")

    def save_comparison_results(self, comparison_df: pd.DataFrame, predictions_df: pd.DataFrame):
        """Сохранение результатов сравнения"""
        print(f"\n💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ...")

        # Детальные результаты сравнения
        detailed_results = comparison_df[[
            'Id экзамена', 'Id вопроса', '№ вопроса',
            'Оценка экзаменатора', 'predicted_score', 'predicted_class',
            'confidence', 'match', 'max_score'
        ]]

        detailed_results.to_csv('detailed_comparison_results.csv', index=False, sep=';')
        print("   Детальные результаты сравнения: 'detailed_comparison_results.csv'")

        # Все предсказания
        predictions_df.to_csv('all_predictions.csv', index=False, sep=';')
        print("   Все предсказания: 'all_predictions.csv'")

        # Сводная статистика
        summary = {
            'total_records': len(comparison_df),
            'overall_accuracy': comparison_df['match'].mean() * 100,
            'mean_confidence': comparison_df['confidence'].mean(),
            'mean_absolute_error': abs(comparison_df['predicted_score'] - comparison_df['Оценка экзаменатора']).mean()
        }

        # Добавляем точность по вопросам
        for question_num in range(1, 5):
            question_data = comparison_df[comparison_df['№ вопроса'] == question_num]
            if len(question_data) > 0:
                accuracy = question_data['match'].mean() * 100
                summary[f'accuracy_question_{question_num}'] = accuracy

        summary_df = pd.DataFrame([summary])
        summary_df.to_csv('comparison_summary.csv', index=False, sep=';')
        print("   Сводная статистика: 'comparison_summary.csv'")

        print(f"\n✅ РЕЗУЛЬТАТЫ СОХРАНЕНЫ!")


def main():
    """Основная функция для сравнения предсказаний"""

    # Инициализация модели
    print("🚀 ЗАГРУЗКА МОДЕЛИ ДЛЯ СРАВНЕНИЯ...")
    comparator = PredictionComparator("./fine_tuned_rubert_base")

    # Шаг 1: Предсказание оценок для clear_test.csv
    predictions_df = comparator.predict_test_set("clear_test.csv")

    if predictions_df.empty:
        print("❌ Не удалось сделать предсказания")
        return

    # Шаг 2: Загрузка правильных оценок
    true_marks_df = comparator.load_true_marks("clear_test_with_marks.csv")

    if true_marks_df.empty:
        print("❌ Не удалось загрузить правильные оценки")
        return

    # Шаг 3: Сравнение предсказаний с правильными оценками
    comparison_df = comparator.compare_predictions(predictions_df, true_marks_df)

    if comparison_df is None or comparison_df.empty:
        print("❌ Не удалось сравнить предсказания")
        return

    # Шаг 4: Генерация детального отчета
    comparator.generate_detailed_report(comparison_df)

    # Шаг 5: Сохранение результатов
    comparator.save_comparison_results(comparison_df, predictions_df)

    print(f"\n🎉 СРАВНЕНИЕ ЗАВЕРШЕНО!")
    print(f"📊 Обработано {len(comparison_df)} записей")


if __name__ == "__main__":
    main()
