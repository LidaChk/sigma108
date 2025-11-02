# analyze_training_distribution.py
import pandas as pd
import matplotlib.pyplot as plt


def analyze_training_distribution():
    """Анализ распределения оценок в обучающих данных"""

    train_df = pd.read_csv('clear_test_with_marks.csv', sep=',')

    print("📊 РАСПРЕДЕЛЕНИЕ В ОБУЧАЮЩИХ ДАННЫХ:")
    print("=" * 50)

    for question_num in range(1, 5):
        question_data = train_df[train_df['№ вопроса'] == question_num]
        score_counts = question_data['Оценка экзаменатора'].value_counts().sort_index()

        print(f"\n❓ Вопрос {question_num}:")
        for score, count in score_counts.items():
            percentage = (count / len(question_data)) * 100
            print(f"   Оценка {score}: {count} примеров ({percentage:.1f}%)")

        # Визуализация
        plt.figure(figsize=(10, 6))
        score_counts.plot(kind='bar', color=['red', 'orange', 'green'])
        plt.title(f'Распределение оценок для вопроса {question_num}')
        plt.xlabel('Оценка')
        plt.ylabel('Количество примеров')
        plt.savefig(f'question_{question_num}_distribution.png', dpi=300, bbox_inches='tight')
        print(f"   💾 Визуализация сохранена в 'question_{question_num}_distribution.png'")


analyze_training_distribution()