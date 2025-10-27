import pandas as pd
from bs4 import BeautifulSoup
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

# -----------------------------
# 1. Предобработка данных
# -----------------------------
def clean_html(text):
    if pd.isna(text):
        return ""
    return BeautifulSoup(str(text), "html.parser").get_text(separator=" ").strip()

def format_prompt(row):
    question_number = row.get("№ вопроса", "")
    question_text = clean_html(row.get("Текст вопроса", ""))
    answer_text = str(row.get("Транскрибация ответа", ""))
    max_score = row.get("Оценка экзаменатора", "")

    prompt = (
        f"Вопрос №{question_number}: {question_text}\n"
        f"Ответ: {answer_text}\n"
        f"Предскажи оценку экзаменатора (число):"
    )
    return {"text": prompt, "labels": str(max_score)}

# Загружаем CSV
df = pd.read_csv("data.csv")

# Получаем 50% строк
# df_subset = df.sample(frac=0.5, random_state=42)  # случайные 50%
# Или первые 50%
df_subset = df.iloc[:len(df)//2]

dataset = Dataset.from_pandas(df_subset.apply(format_prompt, axis=1, result_type="expand"))

# -----------------------------
# 2. Загрузка модели и токенизатора
# -----------------------------
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": device},
    dtype=torch.float16 if device=="mps" else torch.float32,
    low_cpu_mem_usage=True
)

print("Модель загружена на устройство:", next(model.parameters()).device)
print("Dtype параметров:", next(model.parameters()).dtype)

# -----------------------------
# 3. Настройка LoRA
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# -----------------------------
# 4. Токенизация (исправляем labels)
# -----------------------------
def tokenize_fn(examples):
    # токенизируем весь prompt
    encodings = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    # токенизируем число оценки
    label_ids = tokenizer(
        examples["labels"],
        truncation=True,
        padding="max_length",
        max_length=5
    )["input_ids"]

    # создаем labels для LM: -100 для всех токенов кроме числа
    batch_labels = []
    for i in range(len(encodings["input_ids"])):
        ids = encodings["input_ids"][i]
        lbl = [-100] * len(ids)
        # вставляем число в конец
        lbl[-len(label_ids[i]):] = label_ids[i]
        batch_labels.append(lbl)

    encodings["labels"] = batch_labels
    return encodings


tokenized_dataset = dataset.map(tokenize_fn, batched=True)

# -----------------------------
# 5. TrainingArguments с логированием
# -----------------------------
training_args = TrainingArguments(
    output_dir="./qwen-lora-m1",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=False,  # на MPS не поддерживается через Trainer
    save_total_limit=1,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100
)

# -----------------------------
# 6. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# -----------------------------
# 7. Запуск тренировки
# -----------------------------
trainer.train()

# -----------------------------
# 8. Сохранение модели
# -----------------------------
model.save_pretrained("./qwen-lora-m1")
tokenizer.save_pretrained("./qwen-lora-m1")
