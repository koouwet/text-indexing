import json
import requests

queries = [
    "Искусственный интеллект",
    "Машинное обучение",
    "Нейронная сеть",
    "Глубокое обучение",
    "Компьютерное зрение",
    "Обработка естественного языка",
    "Обучение с подкреплением",
    "Трансформер (архитектура)",
    "Градиентный бустинг",
    "Рекуррентная нейронная сеть",
    "Сверточная нейронная сеть",
    "Алгоритм",
    "Компьютерная программа",
    "Информатика",
    "Программное обеспечение",
]

corpus = []
seen_titles = set()
doc_id = 1

# Википедия требует описательный User-Agent с контактами
headers = {
    "User-Agent": "UniversityLabSearchBot/1.0 (student_project@mail.ru) python-requests"
}

session = requests.Session()
session.headers.update(headers)

print("Сбор реальных статей из русской Википедии через API...")

for query in queries:
  if len(corpus) >= 500:
    break

  search_url = "https://ru.wikipedia.org/w/api.php"
  params = {
      "action": "query",
      "list": "search",
      "srsearch": query,
      "format": "json",
      "srlimit": 50,
  }

  try:
    response = session.get(search_url, params=params, timeout=10)
    if response.status_code != 200:
      print(f"Ошибка HTTP {response.status_code} для запроса {query}")
      continue

    data = response.json()
    search_results = data.get("query", {}).get("search", [])

    for item in search_results:
      title = item["title"]
      if title in seen_titles:
        continue

      extract_params = {
          "action": "query",
          "prop": "extracts",
          "explaintext": True,
          "titles": title,
          "format": "json",
      }
      ext_response = session.get(search_url, params=extract_params, timeout=10)
      if ext_response.status_code != 200:
        continue

      ext_data = ext_response.json()
      pages = ext_data.get("query", {}).get("pages", {})

      for page_id, page_info in pages.items():
        if page_id == "-1":
          continue
        text = page_info.get("extract", "")
        if len(text.strip()) > 300:
          corpus.append({
              "id": f"doc_{doc_id:03d}",
              "title": title,
              "summary": text.strip()[:200] + "...",
              "text": text.strip(),
          })
          seen_titles.add(title)
          doc_id += 1
          print(f"[{len(corpus)}/500] Добавлено: {title}")

          if len(corpus) >= 500:
            break
      if len(corpus) >= 500:
        break
  except Exception as e:
    print(f"Ошибка при запросе {query}: {e}")

with open("corpus.json", "w", encoding="utf-8") as f:
  json.dump(corpus, f, ensure_ascii=False, indent=4)

print(
    f"Готово! Собрано реальных документов: {len(corpus)}, сохранено в corpus.json"
)