import json
from datasets import load_dataset

print("Загрузка данных из Википедии...")
# Загружаем русскую Википедию в потоковом режиме
dataset = load_dataset(
    "wikimedia/wikipedia", "20231101.ru", split="train", streaming=True
)

keywords = [
    "искусственный интеллект",
    "машинное обучение",
    "нейронная сеть",
    "алгоритм",
    "глубокое обучение",
    "робототехника",
    "компьютерное зрение",
    "модель",
]

corpus = []
doc_id = 1

for item in dataset:
  text = item.get("text", "")
  title = item.get("title", "")

  # Фильтрация по ключевым словам и минимальной длине (> 500 символов)
  combined_text = (title + " " + text).lower()
  if any(kw in combined_text for kw in keywords) and len(text.strip()) > 500:
    summary = text.strip()[:200] + "..." # генерация краткого содержания

    doc = {
        "id": f"doc_{doc_id:03d}",
        "title": title,
        "summary": summary,
        "text": text.strip(),
    }
    corpus.append(doc)
    doc_id += 1

    if len(corpus) >= 500:
      break

# Сохранение в JSON
with open("corpus.json", "w", encoding="utf-8") as f:
  json.dump(corpus, f, ensure_ascii=False, indent=4)

print(f"Готово! Собрано документов: {len(corpus)}, сохранено в corpus.json")