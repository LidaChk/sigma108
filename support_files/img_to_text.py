import re
import json
import ollama
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from html import unescape


INPUT_CSV = "data.csv"
OUTPUT_CSV = "data_with_captions.csv"
CACHE_FILE = "image_captions_cache.json"


def extract_image_url(text: str):
    """Возвращает прямую ссылку или извлекает из <img src="...">."""
    if not text:
        return None
    text = text.strip()
    # Если это прямая ссылка
    if text.startswith("http"):
        return text
    # Если это HTML с <img>
    match = re.search(r'<img\s+[^>]*src="([^"]+)"', text)
    return match.group(1) if match else None


def clean_question_text(html_text: str):
    """
    Убирает HTML-теги, кроме <img>, и удаляет последние 2 строки с <br>,
    которые содержат вопросы про личное мнение.
    """
    if not html_text:
        return ""

    # Убираем переносы строк и пробелы
    text = re.sub(r'\s+', ' ', html_text)

    # Сохраняем <img>, удаляем остальное HTML
    text = re.sub(r'<(?!img\b)[^>]*>', '', text)

    # Разделяем по <br> и убираем последние 2 блока (личные вопросы)
    parts = re.split(r'<br\s*/?>', text)
    if len(parts) > 2:
        text = '<br>'.join(parts[:-2])

    return unescape(text).strip()


def load_cache():
    if Path(CACHE_FILE).exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def describe_image(image_url: str, question_text: str):
    """Генерирует описание изображения с учётом текста вопроса."""
    prompt = (
        f"Вот текст задания для описания картинки:\n\n"
        f"{question_text}\n\n"
        f"Используя этот контекст, опиши изображение ({image_url}) подробно, "
        f"без упоминания экзамена. Расскажи, что изображено, где происходит действие, "
        f"кто присутствует, что они делают, какие эмоции, сезон и атмосфера."
    )

    try:
        response = ollama.chat(
            model="qwen2.5vl:7b",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.get("message", {}).get("content", "").strip()
        return content

    except Exception as e:
        print(f" Ошибка при обработке {image_url}: {e}")
        return None


def process_csv():
    try:
        df = pd.read_csv(INPUT_CSV, sep=',', encoding='utf-8')
    except pd.errors.ParserError:
        df = pd.read_csv(INPUT_CSV, sep=';', encoding='utf-8')

    if len(df.columns) == 1:
        print(" Похоже, CSV содержит один столбец. Проверь разделитель.")
        print(df.head(5))
        return

    cache = load_cache()
    df['Описание_картинки'] = None

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Обработка строк"):
        if str(row.get('№ вопроса')).strip() == '4':
            html_text = str(row.get('Картинка из вопроса', ''))
            image_url = extract_image_url(html_text)

            if not image_url:
                continue

            question_text = clean_question_text(str(row.get('Текст вопроса', '')))

            if image_url in cache:
                description = cache[image_url]
            else:
                description = describe_image(image_url, question_text)
                if description:
                    cache[image_url] = description
                    save_cache(cache)

            if description:
                df.at[i, 'Описание_картинки'] = description

                # Берём исходный текст вопроса
                original_text = str(row.get('Текст вопроса', ''))

                # Убираем <img src="...">
                cleaned_text = re.sub(r'<img[^>]+>', '', original_text)

                # Добавляем расшифровку изображения
                df.at[i, 'Текст вопроса'] = f"{cleaned_text.strip()}\n\nРасшифровка изображения: {description}"


    df.to_csv(OUTPUT_CSV, index=False)
    save_cache(cache)
    print(f"\n Обработка завершена. Сохранено в {OUTPUT_CSV}")
    print(f"Кэш сохранён в {CACHE_FILE} (описаний: {len(cache)})")


if __name__ == "__main__":
    process_csv()
