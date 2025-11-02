import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

# 🔧 НАСТРОЙКИ ДЛЯ M1 PRO
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🎯 Используемое устройство: {DEVICE}")


# 📁 Класс для датасета
class ExamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 📊 Функция для вычисления метрик
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    f1_per_class = f1_score(labels, predictions, average=None)

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_class_0': f1_per_class[0],
        'f1_class_1': f1_per_class[1],
        'f1_class_2': f1_per_class[2],
    }


# 🎯 Кастомный Trainer с Weighted Loss
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        if hasattr(self.model, 'device'):
            self.class_weights = self.class_weights.to(self.model.device)
        else:
            self.class_weights = self.class_weights.to(DEVICE)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def train_rubert_base():
    """Обучение более мощной модели RuBERT-base"""

    print("🚀 ЗАПУСК ОБУЧЕНИЯ RuBERT-base")
    print("=" * 50)

    # 🔧 НАСТРОЙКИ ДЛЯ RuBERT-base
    MODEL_NAME = "ai-forever/ruBert-base"  # ✅ Более мощная модель
    BATCH_SIZE = 4  # ✅ Уменьшаем batch size из-за большего размера модели
    MAX_LENGTH = 256  # ✅ Уменьшаем длину для экономии памяти
    LEARNING_RATE = 1e-5  # ✅ Меньше learning rate для большей модели
    EPOCHS = 8  # ✅ Меньше эпох (модель обучается быстрее)
    WARMUP_RATIO = 0.1

    # 📁 ЗАГРУЗКА ДАННЫХ
    print("📥 Загрузка данных...")
    train_df = pd.read_csv('train_text_dataset.csv', sep=';')
    val_df = pd.read_csv('val_text_dataset.csv', sep=';')

    print(f"📊 Данные для обучения:")
    print(f"  Обучающая выборка: {len(train_df)} примеров")
    print(f"  Валидационная выборка: {len(val_df)} примеров")

    # Анализ распределения
    train_counts = train_df['Оценка экзаменатора'].value_counts().sort_index()
    print("\n📈 Распределение в тренировочных данных:")
    for score, count in train_counts.items():
        percentage = (count / len(train_df)) * 100
        print(f"  Оценка {score}: {count} примеров ({percentage:.1f}%)")

    # 🎯 ВЕСА КЛАССОВ (оптимизированные для rubert-base)
    class_weights = np.array([3.5, 0.5, 0.9])  # ✅ Более агрессивные веса для класса 0
    print(f"\n⚖️ Агрессивные веса классов: {class_weights}")

    # 🤖 ЗАГРУЗКА RuBERT-base
    print("🤖 Загрузка RuBERT-base...")
    print("⚠️  Внимание: Модель большая (~700МБ), загрузка может занять время...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Загружаем модель с автоматическим определением устройства
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label={0: "оценка_0", 1: "оценка_1", 2: "оценка_2"},
        label2id={"оценка_0": 0, "оценка_1": 1, "оценка_2": 2},
        ignore_mismatched_sizes=True
    )

    # 📚 СОЗДАНИЕ ДАТАСЕТОВ
    print("📚 Создание датасетов...")
    train_dataset = ExamDataset(
        texts=train_df['текст_для_обучения'].values,
        labels=train_df['Оценка экзаменатора'].values,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    val_dataset = ExamDataset(
        texts=val_df['текст_для_обучения'].values,
        labels=val_df['Оценка экзаменатора'].values,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    # ⚙️ ПАРАМЕТРЫ ОБУЧЕНИЯ ДЛЯ RuBERT-base
    training_args = TrainingArguments(
        output_dir='./rubert_base_results',
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        logging_dir='./rubert_base_logs',
        logging_steps=100,
        eval_steps=100,
        save_steps=200,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to=None,
        fp16=False,  # ❌ Лучше отключить для стабильности на M1
        dataloader_pin_memory=False,
        save_total_limit=3,
        no_cuda=True if DEVICE == "mps" else False,
        remove_unused_columns=False,
        gradient_accumulation_steps=2,  # ✅ Накопление градиентов для эффективного batch size = 8
        dataloader_num_workers=0,
    )

    # 🎯 СОЗДАНИЕ TRAINER
    print("🎯 Создание trainer для RuBERT-base...")
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # 🚀 ЗАПУСК ОБУЧЕНИЯ
    print("🚀 Запуск обучения RuBERT-base...")
    print(f"🔧 КОНФИГУРАЦИЯ ДЛЯ RuBERT-base:")
    print(f"   Устройство: {DEVICE}")
    print(f"   Модель: {MODEL_NAME}")
    print(f"   Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * 2} с gradient accumulation)")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Макс. длина: {MAX_LENGTH}")
    print(f"   Агрессивные веса класса 0: {class_weights[0]}")

    if DEVICE == "mps":
        model = model.to(DEVICE)
        print(f"   Модель перемещена на MPS")

    print(f"\n🎯 ОЖИДАЕМЫЕ УЛУЧШЕНИЯ:")
    print(f"   Accuracy: 75.1% → 80-85%")
    print(f"   F1-macro: 69.8% → 75-80%")
    print(f"   Class 0 F1: 48.8% → 60-70%")

    print(f"\n⏳ Ожидаемое время обучения: 2-4 часа")
    print(f"💾 Память: ~4-6GB RAM")

    # 🏁 ЗАПУСК ОБУЧЕНИЯ
    train_result = trainer.train()

    # 💾 СОХРАНЕНИЕ МОДЕЛИ
    print("💾 Сохранение модели RuBERT-base...")
    trainer.save_model("./fine_tuned_rubert_base")
    tokenizer.save_pretrained("./fine_tuned_rubert_base")

    # 📊 ОЦЕНКА РЕЗУЛЬТАТОВ
    print("📊 Оценка RuBERT-base...")
    eval_results = trainer.evaluate()

    print("\n🎯 РЕЗУЛЬТАТЫ RuBERT-base:")
    print("=" * 50)
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    # 📈 ДЕТАЛЬНЫЙ ОТЧЕТ
    print("\n📈 ДЕТАЛЬНЫЙ ОТЧЕТ ПО КЛАССАМ:")
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = val_df['Оценка экзаменатора'].values

    print(classification_report(true_labels, pred_labels,
                                target_names=['Оценка 0', 'Оценка 1', 'Оценка 2'],
                                digits=4))

    # 📊 ВИЗУАЛИЗАЦИЯ СРАВНЕНИЯ
    plot_comparison_with_previous(true_labels, pred_labels, eval_results)

    return trainer, eval_results


def plot_comparison_with_previous(true_labels, pred_labels, eval_results):
    """Сравнение результатов RuBERT-base с предыдущей моделью"""
    try:
        # Результаты предыдущей модели (rubert-tiny2)
        previous_results = {
            'accuracy': 0.7510,
            'f1_macro': 0.6985,
            'f1_class_0': 0.4880,
            'f1_class_1': 0.7640,
            'f1_class_2': 0.8435
        }

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('📊 Сравнение: RuBERT-base vs RuBERT-tiny2', fontsize=16, fontweight='bold')

        # 1. Confusion Matrix новой модели
        ax1 = axes[0, 0]
        cm = confusion_matrix(true_labels, pred_labels)
        im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax1.set_title('RuBERT-base: Confusion Matrix', fontweight='bold')
        plt.colorbar(im, ax=ax1)
        tick_marks = np.arange(3)
        ax1.set_xticks(tick_marks)
        ax1.set_yticks(tick_marks)
        ax1.set_xticklabels(['0', '1', '2'])
        ax1.set_yticklabels(['0', '1', '2'])
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')

        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            ax1.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontweight='bold')

        # 2. Сравнение основных метрик
        ax2 = axes[0, 1]
        metrics = ['Accuracy', 'F1 Macro']
        previous_values = [previous_results['accuracy'], previous_results['f1_macro']]
        current_values = [eval_results['eval_accuracy'], eval_results['eval_f1_macro']]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax2.bar(x - width / 2, previous_values, width, label='RuBERT-tiny2',
                        color='lightblue', edgecolor='black', alpha=0.8)
        bars2 = ax2.bar(x + width / 2, current_values, width, label='RuBERT-base',
                        color='lightgreen', edgecolor='black', alpha=0.8)

        ax2.set_title('Сравнение основных метрик', fontweight='bold')
        ax2.set_ylabel('Score')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        # Добавляем значения и улучшения
        for i, (prev, curr) in enumerate(zip(previous_values, current_values)):
            improvement = curr - prev
            color = 'green' if improvement > 0 else 'red'
            ax2.text(i, max(prev, curr) + 0.02, f'+{improvement:.3f}',
                     ha='center', va='bottom', fontweight='bold', color=color)

        # 3. Сравнение F1 по классам
        ax3 = axes[1, 0]
        classes = ['Оценка 0', 'Оценка 1', 'Оценка 2']
        previous_f1 = [previous_results['f1_class_0'], previous_results['f1_class_1'], previous_results['f1_class_2']]
        current_f1 = [eval_results['eval_f1_class_0'], eval_results['eval_f1_class_1'], eval_results['eval_f1_class_2']]

        x = np.arange(len(classes))

        bars1 = ax3.bar(x - width / 2, previous_f1, width, label='RuBERT-tiny2',
                        color='lightblue', edgecolor='black', alpha=0.8)
        bars2 = ax3.bar(x + width / 2, current_f1, width, label='RuBERT-base',
                        color='lightgreen', edgecolor='black', alpha=0.8)

        ax3.set_title('Сравнение F1 по классам', fontweight='bold')
        ax3.set_ylabel('F1 Score')
        ax3.set_xticks(x)
        ax3.set_xticklabels(classes)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)

        # Добавляем улучшения
        for i, (prev, curr) in enumerate(zip(previous_f1, current_f1)):
            improvement = curr - prev
            color = 'green' if improvement > 0 else 'red'
            ax3.text(i, max(prev, curr) + 0.02, f'+{improvement:.3f}',
                     ha='center', va='bottom', fontweight='bold', color=color)

        # 4. Информация о модели
        ax4 = axes[1, 1]
        ax4.axis('off')

        total_improvement = (eval_results['eval_accuracy'] - previous_results['accuracy'] +
                             eval_results['eval_f1_macro'] - previous_results['f1_macro']) / 2

        model_info = [
            f"📊 СРАВНЕНИЕ МОДЕЛЕЙ:",
            f"",
            f"RuBERT-tiny2:",
            f"  • Параметров: ~30M",
            f"  • Accuracy: {previous_results['accuracy']:.3f}",
            f"  • F1 Macro: {previous_results['f1_macro']:.3f}",
            f"  • Class 0 F1: {previous_results['f1_class_0']:.3f}",
            f"",
            f"RuBERT-base:",
            f"  • Параметров: ~178M",
            f"  • Accuracy: {eval_results['eval_accuracy']:.3f}",
            f"  • F1 Macro: {eval_results['eval_f1_macro']:.3f}",
            f"  • Class 0 F1: {eval_results['eval_f1_class_0']:.3f}",
            f"",
            f"📈 ОБЩЕЕ УЛУЧШЕНИЕ: {total_improvement:.3f}",
        ]

        for i, text in enumerate(model_info):
            if "СРАВНЕНИЕ" in text or "ОБЩЕЕ УЛУЧШЕНИЕ" in text:
                ax4.text(0.1, 0.95 - i * 0.05, text, transform=ax4.transAxes,
                         fontsize=11, verticalalignment='top', fontweight='bold',
                         color='darkblue')
            elif "RuBERT-base" in text:
                ax4.text(0.1, 0.95 - i * 0.05, text, transform=ax4.transAxes,
                         fontsize=10, verticalalignment='top', fontweight='bold',
                         color='darkgreen')
            else:
                ax4.text(0.1, 0.95 - i * 0.05, text, transform=ax4.transAxes,
                         fontsize=10, verticalalignment='top', fontweight='normal')

        plt.tight_layout()
        plt.savefig('rubert_base_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("📊 Визуализация сравнения сохранена в 'rubert_base_comparison.png'")
        plt.close()

    except Exception as e:
        print(f"⚠️ Не удалось создать визуализацию сравнения: {e}")


def check_memory_usage():
    """Проверка использования памяти"""
    import psutil
    memory = psutil.virtual_memory()
    print(f"💾 Использование памяти: {memory.percent}%")
    print(f"   Доступно: {memory.available / 1024 / 1024 / 1024:.1f} GB")
    print(f"   Всего: {memory.total / 1024 / 1024 / 1024:.1f} GB")


if __name__ == "__main__":
    try:
        # 🔍 ПРОВЕРКА ПАМЯТИ
        check_memory_usage()

        # 🚀 ОБУЧЕНИЕ RuBERT-base
        trainer, results = train_rubert_base()

        # 📈 АНАЛИЗ РЕЗУЛЬТАТОВ
        previous_accuracy = 0.7510
        previous_f1_macro = 0.6985
        previous_class0_f1 = 0.4880

        accuracy_improvement = results['eval_accuracy'] - previous_accuracy
        f1_improvement = results['eval_f1_macro'] - previous_f1_macro
        class0_improvement = results['eval_f1_class_0'] - previous_class0_f1

        print("\n" + "=" * 60)
        print("✅ ОБУЧЕНИЕ RuBERT-base ЗАВЕРШЕНО!")
        print("=" * 60)

        print(f"\n💾 МОДЕЛЬ СОХРАНЕНА:")
        print(f"   ./fine_tuned_rubert_base")

        print(f"\n📊 ВИЗУАЛИЗАЦИЯ:")
        print(f"   rubert_base_comparison.png")

        print(f"\n🎯 РЕЗУЛЬТАТЫ СРАВНЕНИЯ:")
        print(f"   Accuracy: {previous_accuracy:.3f} → {results['eval_accuracy']:.3f} ({accuracy_improvement:+.3f})")
        print(f"   F1 Macro: {previous_f1_macro:.3f} → {results['eval_f1_macro']:.3f} ({f1_improvement:+.3f})")
        print(f"   Class 0 F1: {previous_class0_f1:.3f} → {results['eval_f1_class_0']:.3f} ({class0_improvement:+.3f})")

        print(f"\n💡 РЕКОМЕНДАЦИИ:")
        if accuracy_improvement > 0.03:
            print(f"   🎉 Отличное улучшение! RuBERT-base значительно лучше.")
        elif accuracy_improvement > 0.01:
            print(f"   ✅ Хорошее улучшение. RuBERT-base показывает лучшие результаты.")
        else:
            print(f"   ⚠️ Улучшение минимально. Возможно, нужно больше данных или другая стратегия.")

        if class0_improvement > 0.05:
            print(f"   🎉 Класс 0 значительно улучшился! +{class0_improvement:.3f}")
        elif class0_improvement > 0.02:
            print(f"   ✅ Класс 0 улучшился. +{class0_improvement:.3f}")
        else:
            print(f"   ⚠️ Класс 0 все еще требует внимания.")

    except Exception as e:
        print(f"❌ Ошибка при обучении RuBERT-base: {e}")
        import traceback

        traceback.print_exc()

        # Если проблема с памятью
        if "CUDA out of memory" in str(e) or "memory" in str(e).lower():
            print(f"\n💡 СОВЕТ: Попробуйте уменьшить BATCH_SIZE до 2 или MAX_LENGTH до 128")

