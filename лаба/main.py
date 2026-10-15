import json
import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

# Обходим проблему с SSL-сертификатами на Windows
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

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
  url = f"http://export.arxiv.org/api/query?search_query=cat:{cat}&start=0&max_results=130"
  
  # Передаем контекст безопасности
  data = urllib.request.urlopen(url, context=ctx).read()
  root = ET.fromstring(data)

  ns = {"atom": "http://www.w3.org/2005/Atom"}

  for entry in root.findall("atom:entry", ns):
    title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
    summary_raw = (
        entry.find("atom:summary", ns).text.strip().replace("\n", " ")
    )

    doc = {
        "id": f"doc_{doc_id:03d}",
        "title": f"[{cat_name}] {title}",
        "summary": summary_raw[:200] + "...",
        "text": f"Научное исследование в области {cat_name}. Тема: {title}. Полное описание работы: {summary_raw}",
    }
    corpus.append(doc)
    doc_id += 1

corpus = corpus[:500]

with open("corpus.json", "w", encoding="utf-8") as f:
  json.dump(corpus, f, ensure_ascii=False, indent=4)

print(
    f"Готово! В corpus.json сохранен 100% целевой корпус из {len(corpus)} научных статей по ИИ."
)