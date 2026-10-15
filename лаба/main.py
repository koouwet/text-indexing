import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

# Профильные категории arXiv для ИИ
CATEGORIES = [
    ("cs.AI", "Искусственный интеллект"),
    ("cs.LG", "Машинное обучение"),
    ("cs.CV", "Компьютерное зрение"),
    ("cs.CL", "Обработка естественного языка"),
]

corpus = []
doc_id = 1

print("Скачиваем статьи напрямую с arXiv API...")

for cat, cat_name in CATEGORIES:
  # Запрашиваем по 130 статей на каждую подтему ИИ
  url = f"http://export.arxiv.org/api/query?search_query=cat:{cat}&start=0&max_results=130"
  data = urllib.request.urlopen(url).read()
  root = ET.fromstring(data)

  ns = {"atom": "http://www.w3.org/2005/Atom"}

  for entry in root.findall("atom:entry", ns):
    title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
    summary_raw = (
        entry.find("atom:summary", ns).text.strip().replace("\n", " ")
    )

    # Формируем структуру документа
    doc = {
        "id": f"doc_{doc_id:03d}",
        "title": f"[{cat_name}] {title}",
        "summary": summary_raw[:200] + "...",
        "text": f"Научное исследование в области {cat_name}. Тема: {title}. Полное описание работы: {summary_raw}",
    }
    corpus.append(doc)
    doc_id += 1

# Сохраняем ровно 500 документов (или столько, сколько собрали)
corpus = corpus[:500]

with open("corpus.json", "w", encoding="utf-8") as f:
  json.dump(corpus, f, ensure_ascii=False, indent=4)

print(
    f"Готово! В corpus.json сохранен 100% целевой корпус из {len(corpus)} научных статей по ИИ."
)