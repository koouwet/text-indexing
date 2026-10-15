import argparse
import json
import math
import re
from collections import defaultdict
import pymorphy3

morph = pymorphy3.MorphAnalyzer()


def preprocess(text):
  """Токенизация, приведение к нижнему регистру и лемматизация"""
  # Исправленный паттерн для поиска русских и английских слов
  words = re.findall(r"[a-zа-яё]+", text.lower())
  return [morph.parse(w)[0].normal_form for w in words]


def build_index():
  print("Режим обучения: построение инвертированного индекса...")
  with open("corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

  inverted_index = defaultdict(list)
  doc_lengths = {}
  total_length = 0

  for doc in corpus:
    doc_id = doc["id"]
    # Объединяем заголовок и текст для индексации
    full_text = doc["title"] + " " + doc["text"]
    tokens = preprocess(full_text)

    doc_lengths[doc_id] = len(tokens)
    total_length += len(tokens)

    # Считаем частоту терминов в документе
    term_freqs = defaultdict(int)
    for token in tokens:
      term_freqs[token] += 1

    for token, freq in term_freqs.items():
      inverted_index[token].append((doc_id, freq))

  avgdl = total_length / len(corpus) if corpus else 0

  # Сохраняем индекс и метаданные
  data = {
      "index": dict(inverted_index),
      "doc_lengths": doc_lengths,
      "avgdl": avgdl,
      "total_docs": len(corpus),
  }

  with open("inverted_index.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

  print("Индекс успешно построен и сохранен в inverted_index.json!")


def search_bm25(query, top_k=5):
  with open("inverted_index.json", "r", encoding="utf-8") as f:
    data = json.load(f)

  with open("corpus.json", "r", encoding="utf-8") as f:
    corpus = {doc["id"]: doc for doc in json.load(f)}

  index = data["index"]
  doc_lengths = data["doc_lengths"]
  avgdl = data["avgdl"]
  N = data["total_docs"]

  query_tokens = preprocess(query)
  scores = defaultdict(float)

  k1 = 1.5
  b = 0.75

  for q in query_tokens:
    if q not in index:
      continue

    postings = index[q]  # список пар (doc_id, freq)
    n_q = len(postings)  # количество документов, содержащих термин

    # Вычисление IDF для BM25
    idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)
    if idf < 0:
      idf = 0

    for doc_id, freq in postings:
      dl = doc_lengths[doc_id]
      # Формула BM25
      numerator = freq * (k1 + 1)
      denominator = freq + k1 * (1 - b + b * (dl / avgdl))
      scores[doc_id] += idf * (numerator / denominator)

  # Сортируем документы по убыванию оценки BM25
  sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

  print(f"\nРезультаты поиска по запросу: '{query}'")
  results = []
  for i, (doc_id, score) in enumerate(sorted_docs[:top_k], 1):
    doc = corpus[doc_id]
    results.append(doc_id)
    print(f"{i}. [{doc_id}] {doc['title']} (BM25: {score:.4f})")
    print(f"   {doc['summary']}\n")

  if not results:
    print("Ничего не найдено по вашему запросу.")

  return results


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Лексический поиск BM25")
  parser.add_argument(
      "--mode", choices=["train", "search"], required=True, help="Режим работы"
  )
  parser.add_argument("--query", type=str, help="Поисковый запрос")

  args = parser.parse_args()

  if args.mode == "train":
    build_index()
  elif args.mode == "search":
    if not args.query:
      print("Укажите поисковый запрос с помощью аргумента --query 'ваш запрос'")
    else:
      search_bm25(args.query)