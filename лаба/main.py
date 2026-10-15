import json
from datasets import load_dataset

print("Загрузка данных из Википедии...")
dataset = load_dataset(
    "wikimedia/wikipedia", "20231101.ru", split="train", streaming=True
)

# Точные ключевые фразы про ИИ
strict_keywords = [
    "искусственный интеллект",
    "машинное обучение",
    "нейронная сеть",
    "нейронных сетей",
    "глубокое обучение",
    "компьютерное зрение",
    "обработка естественного языка",
    "обучение с подкреплением",
    "большая языковая модель",
    "трансформер (архитектура",
]

corpus = []
doc_id = 1

for item in dataset:
  text = item.get("text", "")
  title = item.get("title", "")

  combined_text = (title + " " + text).lower()

  # Проверяем, что статья содержит именно ИИ-термины и достаточно длинная
  if (
      any(kw in combined_text for kw in strict_keywords)
      and len(text.strip()) > 500
  ):
    summary = text.strip()[:200] + "..."  # генерация краткого содержания

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

with open("corpus.json", "w", encoding="utf-8") as f:
  json.dump(corpus, f, ensure_ascii=False, indent=4)

print(f"Готово! Собрано релевантных документов: {len(corpus)}")