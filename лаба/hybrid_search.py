import argparse
import json
import math
import pickle
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pymorphy3

morph = pymorphy3.MorphAnalyzer()


def preprocess_text(text):
  words = re.findall(r"[a-zа-яё]+", text.lower())
  return [morph.parse(w)[0].normal_form for w in words]


def search_hybrid(query, top_k=5):
  # 1. Загружаем данные BM25
  with open("inverted_index.json", "r", encoding="utf-8") as f:
    bm25_data = json.load(f)

  # 2. Загружаем векторную модель
  with open("vector_model.pkl", "rb") as f:
    vector_pack = pickle.load(f)

  # 3. Загружаем корпус
  with open("corpus.json", "r", encoding="utf-8") as f:
    corpus = {doc["id"]: doc for doc in json.load(f)}

  # --- Шаг А: Считаем BM25 ранг ---
  index = bm25_data["index"]
  doc_lengths = bm25_data["doc_lengths"]
  avgdl = bm25_data["avgdl"]
  N = bm25_data["total_docs"]

  query_tokens = preprocess_text(query)
  bm25_scores = {}
  k1, b = 1.5, 0.75

  for q in query_tokens:
    if q not in index:
      continue
    postings = index[q]
    n_q = len(postings)
    idf = max(0, math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0))
    for doc_id, freq in postings:
      dl = doc_lengths[doc_id]
      score = idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * (dl / avgdl)))
      bm25_scores[doc_id] = bm25_scores.get(doc_id, 0.0) + score

  sorted_bm25 = sorted(bm25_scores.keys(), key=lambda x: bm25_scores[x], reverse=True)
  bm25_ranks = {doc_id: rank + 1 for rank, doc_id in enumerate(sorted_bm25)}

  # --- Шаг Б: Считаем Векторный ранг ---
  vectorizer = vector_pack["vectorizer"]
  svd = vector_pack["svd"]
  doc_vectors = vector_pack["data"]["doc_vectors"]
  doc_ids = vector_pack["data"]["doc_ids"]

  proc_query_str = " ".join(query_tokens)
  query_tfidf = vectorizer.transform([proc_query_str])
  query_vector = svd.transform(query_tfidf)
  similarities = cosine_similarity(query_vector, doc_vectors)[0]

  sorted_vec_indices = np.argsort(similarities)[::-1]
  vector_ranks = {doc_ids[idx]: rank + 1 for rank, idx in enumerate(sorted_vec_indices)}

  # --- Шаг В: Объединение через RRF (Reciprocal Rank Fusion) ---
  rrf_k = 60
  all_docs = set(bm25_ranks.keys()).union(set(vector_ranks.keys()))
  rrf_scores = {}

  for doc_id in all_docs:
    rank_bm25 = bm25_ranks.get(doc_id, 1000)
    rank_vec = vector_ranks.get(doc_id, 1000)

    score = (1.0 / (rrf_k + rank_bm25)) + (1.0 / (rrf_k + rank_vec))
    rrf_scores[doc_id] = score

  # Сортировка по итоговой гибридной оценке RRF
  sorted_hybrid = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

  print(f"\nРезультаты гибридного поиска (RRF) по запросу: '{query}'")
  results = []
  for i, (doc_id, score) in enumerate(sorted_hybrid[:top_k], 1):
    doc = corpus[doc_id]
    results.append(doc_id)
    print(f"{i}. [{doc_id}] {doc['title']} (RRF Score: {score:.5f})")
    print(f"   {doc['summary']}\n")

  return results


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Гибридный поиск (BM25 + SVD)")
  parser.add_argument(
      "--query", type=str, required=True, help="Поисковый запрос"
  )
  args = parser.parse_args()

  search_hybrid(args.query)