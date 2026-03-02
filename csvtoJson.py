import pandas as pd
import requests
import json
import time
import os
import re
from tqdm import tqdm
import ollama

# ===== НАСТРОЙКИ =====
CSV_FILE = "genshin_articles.csv"          # ваш CSV файл
OUTPUT_FILE = "genshin_qa.jsonl"           # выходной датасет
PROGRESS_FILE = "processed_articles.txt"   # файл для сохранения прогресса
MODEL_NAME = "deepseek-r1:8b"              # модель DeepSeek в Ollama
QUESTIONS_PER_ARTICLE = 5                   # сколько вопросов генерировать на статью
OLLAMA_URL = "http://localhost:11434/api/generate"
DELAY = 1                                    # задержка между запросами (чтобы не перегружать API)
TEMPERATURE = 0.7                            # температура генерации (0.7 — хороший баланс)
MAX_TOKENS = 30000                             # максимум токенов на ответ
# =====================

def load_progress():
    """Загружает список уже обработанных URL (или индексов) из файла прогресса."""
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f)

def save_progress(identifier):
    """Сохраняет идентификатор обработанной статьи (например, URL)."""
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(identifier + "\n")

def clean_think_tags(text):
    """Удаляет все блоки <think>...</think> из текста."""
    # Удаляем многострочные блоки
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def extract_json(text):
    """
    Извлекает JSON-массив из текста.
    Сначала пробует распарсить весь текст как JSON (после удаления think).
    Если не получается, ищет подстроку между первой '[' и последней ']'.
    """
    # Удаляем think-теги
    text = clean_think_tags(text).strip()
    if not text:
        return None

    # Пробуем распарсить весь текст
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Ищем JSON-массив: от первой '[' до последней ']'
    match = re.search(r'(\[.*\])', text, flags=re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    return None

def generate_qa_for_article(title, text, num_questions=5):
    """
    Отправляет статью в DeepSeek-R1 через библиотеку ollama
    и получает список вопросов-ответов.
    """
    prompt = f"""Ты — эксперт по игре Genshin Impact. Прочитай текст статьи и составь {num_questions} пар "Вопрос" и "Ответ", которые охватывают основную информацию из статьи.
Вопросы должны быть такими, как их мог бы задать игрок. Ответы должны быть точными, краткими и взятыми из текста.

Статья: {title}
{text}

Ответ должен быть ТОЛЬКО в формате JSON: список объектов с ключами "instruction" (вопрос) и "output" (ответ).
Никаких дополнительных пояснений, только JSON. Не используй теги <think>.
"""
    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={
                "temperature": TEMPERATURE,
                num_predict": MAX_TOKENS,  # аналог max_tokens
                #"stop": ["</s>", "User:", "\n\n"]
            }
        )
        raw_response = response['response']

        # Далее всё так же: clean_think_tags, extract_json и т.д.
        qa_list = extract_json(raw_response)
        if qa_list and all(isinstance(item, dict) and "instruction" in item and "output" in item for item in qa_list):
            return qa_list
        else:
            with open("debug_responses.txt", "a", encoding="utf-8") as f:
                f.write(f"=== Статья: {title}\n{raw_response}\n\n")
            return []
    except Exception as e:
        print(f"Ошибка при генерации: {e}")
        return []

def main():
    # Читаем CSV
    df = pd.read_csv(CSV_FILE)
    # Проверим наличие колонок; ожидаем 'Заголовок' и 'Текст'
    if 'Заголовок' not in df.columns or 'Текст' not in df.columns:
        print("Ошибка: CSV должен содержать колонки 'Заголовок' и 'Текст'")
        return

    # Загружаем прогресс
    processed = load_progress()

    # Открываем выходной файл для добавления
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Генерация"):
            # Используем URL как уникальный идентификатор (если есть)
            identifier = row.get('URL', str(idx))
            if identifier in processed:
                continue

            title = row['Заголовок']
            text = row['Текст']

            # Пропускаем слишком короткие тексты
            if len(text) < 100:
                print(f"Статья '{title}' слишком короткая, пропускаем")
                save_progress(identifier)
                continue

            qa_pairs = generate_qa_for_article(title, text, QUESTIONS_PER_ARTICLE)
            if qa_pairs:
                for qa in qa_pairs:
                    out_f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                out_f.flush()
                print(f"✅ Статья '{title}' обработана, добавлено {len(qa_pairs)} примеров")
            else:
                print(f"⚠️ Не удалось сгенерировать для '{title}'")

            save_progress(identifier)
            time.sleep(DELAY)

    print(f"\nГотово! Датасет сохранён в {OUTPUT_FILE}")

if __name__ == "__main__":
    main()