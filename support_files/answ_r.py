import os
import re
import pandas as pd
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datetime import datetime

# ---------------------------
# Конфигурация
# ---------------------------
BASE_MODEL = "Qwen/Qwen3-0.6B"
ADAPTER_PATH = "./qwen-lora-m1/checkpoint-1839"
INPUT_CSV = "clear_test_with_marks.csv"  # здесь есть колонка с правильными ответами
OUTPUT_CSV = "result_with_stats.csv"
LOG_FILE = "inference_log.txt"

MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 8
BATCH_SIZE = 1  # для M1 Pro безопасно

# ---------------------------
# Утилиты
# ---------------------------
def log(msg: str):
    """Пишет лог и на экран, и в файл"""
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")

def clean_html(text):
    if pd.isna(text):
        return ""
    return BeautifulSoup(str(text), "html.parser").get_text(separator=" ").strip()

def build_prompt(row):
    question_number = row.get("№ вопроса", "")
    question_text = clean_html(row.get("Текст вопроса", ""))
    answer_text = str(row.get("Транскрибация ответа", ""))
    return f"Вопрос №{question_number}: {question_text}\nОтвет: {answer_text}\nПредскажи оценку экзаменатора (число):"

def extract_number_from_text(text):
    m = re.search(r"\b([0-2])\b", text)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(-?\d+)", text)
    if m2:
        val = int(m2.group(1))
        return min(max(val, 0), 2)
    return None

# ---------------------------
# Инициализация модели
# ---------------------------
log("Определяем устройство...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
log(f"Device: {device}")

log("Загружаем токенизатор...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

log("Загружаем базовую модель...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map={"": device},
    dtype=torch.float16 if device == "mps" else torch.float32,
    low_cpu_mem_usage=True,
)

try:
    log(f"Загружаем адаптер из {ADAPTER_PATH} ...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, device_map={"": device})
    log("✅ Адаптер успешно применён.")
except Exception as e:
    log(f"⚠️ Не удалось загрузить адаптер: {e}")
    model = base_model

model.eval()
log(f"Модель готова. dtype: {next(model.parameters()).dtype}")

# ---------------------------
# Загрузка данных
# ---------------------------
log(f"Читаем CSV: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, sep=";", encoding="utf-8")
df_subset = df.iloc[:len(df)//30]
prompts = df_subset.apply(build_prompt, axis=1).tolist()
preds = []

# ---------------------------
# Генерация
# ---------------------------
log(f"Начинаем инференс ({len(prompts)} записей)...")
for i in range(0, len(prompts), BATCH_SIZE):
    batch_prompts = prompts[i:i + BATCH_SIZE]
    enc = tokenizer(
        batch_prompts,
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        return_tensors="pt"
    )

    for k, v in enc.items():
        enc[k] = v.to(device)

    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    for j, out in enumerate(outputs):
        text = tokenizer.decode(out, skip_special_tokens=True)
        prompt = batch_prompts[j]
        gen_part = text.split("Предскажи оценку экзаменатора (число):")[-1].strip()
        num = extract_number_from_text(gen_part)

        preds.append({"generated_text": gen_part, "predicted_score": num})
        log(f"[{i + j + 1}/{len(prompts)}] → {gen_part} → {num}")

# ---------------------------
# Сравнение с правильными ответами
# ---------------------------
result = pd.concat([df.reset_index(drop=True), pd.DataFrame(preds)], axis=1)

if "Оценка экзаменатора" in result.columns:
    result["correct"] = result["predicted_score"] == result["Оценка экзаменатора"]
    total_correct = result["correct"].sum()
    total = len(result)
    acc = total_correct / total * 100

    log(f"\n📊 Общая точность: {acc:.2f}% ({total_correct}/{total})")

    per_question = (
        result.groupby("№ вопроса")["correct"]
        .agg(["count", "sum"])
        .reset_index()
    )
    per_question["accuracy_%"] = per_question["sum"] / per_question["count"] * 100
    log("\n📈 Точность по вопросам:")
    for _, row in per_question.iterrows():
        log(f"  Вопрос {row['№ вопроса']}: {row['accuracy_%']:.1f}% ({int(row['sum'])}/{int(row['count'])})")

else:
    log("⚠️ В файле нет колонки 'Оценка экзаменатора' — статистика не рассчитана.")

# ---------------------------
# Сохранение
# ---------------------------
result.to_csv(OUTPUT_CSV, index=False)
log(f"\n✅ Готово. Результат сохранён в {OUTPUT_CSV}")
