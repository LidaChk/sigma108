import pandas as pd
from bs4 import BeautifulSoup

# Загружаем CSV
df = pd.read_csv("data_with_captions.csv")

# Функция для очистки HTML
def clean_html(text):
    if pd.isna(text):
        return text
    # Убираем HTML теги
    soup = BeautifulSoup(text, "html.parser")
    # Получаем текст с сохранением переносов
    cleaned_text = soup.get_text(separator="\n")
    # Убираем лишние пробелы и пустые строки
    lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
    return " ".join(lines)

# Применяем очистку к столбцу "Текст вопроса"
df["Текст вопроса"] = df["Текст вопроса"].apply(clean_html)

# Сохраняем обратно
df.to_csv("data_with_captions_clean.csv", index=False)
