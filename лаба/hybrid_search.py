import argparse
import json
import math
import pickle
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pymorphy3

morph = pymorphy3.MorphAnalyzer()


def preprocess_text(text):
  words = re.findall(r"[a-zа-яё]+", text.lower())
  return [morph.parse(w)[0].normal_form for w in words]


def get_bm25_candidates(query, bm25_data, top_n=20):
  """Лексический поиск кандидатов (BM25)"""
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

  sorted_bm25 = sorted(
      bm25_scores.items(), key=lambda item: item[1], reverse=True
  )[:top_n]
  return [doc_id for doc_id, _ in sorted_bm25]


def get_semantic_candidates(query, neural_data, model, top_n=20):
  """Семантический поиск кандидатов (Косинусное сходство)"""
  doc_vectors = neural_data["doc_vectors"]
  doc_ids = neural_data["doc_ids"]

  query_vector = model.encode([query])
  similarities = cosine_similarity(query_vector, doc_vectors)[0]
  sorted_indices = np.argsort(similarities)[::-1][:top_n]

  return [doc_ids[idx] for idx in sorted_indices]


def calculate_bm25_single(query, doc_id, bm25_data, corpus):
  """Вспомогательная функция для расчета точного BM25 для одного документа"""
  index = bm25_data["index"]
  doc_lengths = bm25_data["doc_lengths"]
  avgdl = bm25_data["avgdl"]
  N = bm25_data["total_docs"]
  k1, b = 1.5, 0.75

  query_tokens = preprocess_text(query)
  score = 0.0
  for q in query_tokens:
    if q in index:
      for d_id, freq in index[q]:
        if d_id == doc_id:
          n_q = len(index[q])
          idf = max(0, math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0))
          dl = doc_lengths[doc_id]
          score += idf * (freq * (k1 + 1)) / (
              freq + k1 * (1 - b + b * (dl / avgdl))
          )
  return score


def calculate_cosine_single(query, doc_id, neural_data, model):
  """Вспомогательная функция для расчета точного косинусного сходства для одного документа"""
  doc_vectors = neural_data["doc_vectors"]
  doc_ids = neural_data["doc_ids"]

  if doc_id not in doc_ids:
    return 0.0

  idx = doc_ids.index(doc_id)
  query_vector = model.encode([query])
  sim = cosine_similarity(query_vector, [doc_vectors[idx]])[0][0]
  return float(sim)


def get_candidate_documents(
    query, bm25_data, neural_data, model, corpus, top_n=20
):
  """Подпункты 1 и 2: Получение кандидатов и вычисление структуры оценок"""
  bm25_cand = get_bm25_candidates(query, bm25_data, top_n=top_n)
  semantic_cand = get_semantic_candidates(query, neural_data, model, top_n=top_n)

  # Объединение без дубликатов
  combined_ids = list(dict.fromkeys(bm25_cand + semantic_cand))

  # Структура данных: ID -> BM25, Cosine
  candidate_scores = []
  for doc_id in combined_ids:
    bm25_sc = calculate_bm25_single(query, doc_id, bm25_data, corpus)
    cos_sc = calculate_cosine_single(query, doc_id, neural_data, model)
    candidate_scores.append({
        "doc_id": doc_id,
        "bm25_score": bm25_sc,
        "cosine_score": cos_sc,
    })

  return candidate_scores


def normalize_scores(candidate_scores):
  """Подпункт 3: Нормализация оценок (Min-Max)"""
  if not candidate_scores:
    return candidate_scores

  bm25_vals = [c["bm25_score"] for c in candidate_scores]
  cos_vals = [c["cosine_score"] for c in candidate_scores]

  min_bm25, max_bm25 = min(bm25_vals), max(bm25_vals)
  min_cos, max_cos = min(cos_vals), max(cos_vals)

  for c in candidate_scores:
    # Нормализация BM25
    if max_bm25 == min_bm25:
      c["norm_bm25"] = 0.0
    else:
      c["norm_bm25"] = (c["bm25_score"] - min_bm25) / (max_bm25 - min_bm25)

    # Нормализация Cosine
    if max_cos == min_cos:
      c["norm_cosine"] = 0.0
    else:
      c["norm_cosine"] = (c["cosine_score"] - min_cos) / (max_cos - min_cos)

  return candidate_scores


def calculate_final_score(norm_bm25, norm_cosine, alpha):
  """Подпункт 4: Расчет итоговой оценки релевантности"""
  return alpha * norm_bm25 + (1.0 - alpha) * norm_cosine


def search_hybrid(query, alpha=0.5, top_k=5):
  """Подпункт 6: Выполнение поиска с линейной комбинацией"""
  with open("corpus.json", "r", encoding="utf-8") as f:
    corpus = {doc["id"]: doc for doc in json.load(f)}

  with open("inverted_index.json", "r", encoding="utf-8") as f:
    bm25_data = json.load(f)

  with open("neural_model.pkl", "rb") as f:
    neural_data = pickle.load(f)

  model = SentenceTransformer(neural_data["model_name"])

  # 1-2. Кандидаты и исходные оценки
  candidates = get_candidate_documents(
      query, bm25_data, neural_data, model, corpus, top_n=20
  )

  # 3. Нормализация
  candidates = normalize_scores(candidates)

  # 4. Расчет финальной оценки
  for c in candidates:
    c["final_score"] = calculate_final_score(
        c["norm_bm25"], c["norm_cosine"], alpha
    )

  # Сортировка по убыванию итоговой оценки
  candidates.sort(key=lambda x: x["final_score"], reverse=True)

  print(
      f"\nРезультаты гибридного поиска (alpha={alpha}) по запросу: '{query}'"
  )
  for i, c in enumerate(candidates[:top_k], 1):
    doc_id = c["doc_id"]
    doc = corpus[doc_id]
    print(f"{i}. [{doc_id}] {doc['title']} (Final Score: {c['final_score']:.4f})")
    print(
        f"   [BM25: {c['bm25_score']:.2f} (norm: {c['norm_bm25']:.2f}) | Cosine:"
        f" {c['cosine_score']:.4f} (norm: {c['norm_cosine']:.2f})]"
    )
    print(f"   Краткое содержание: {doc.get('summary', doc['text'][:150])}")
    print(f"   Текст документа: {doc['text'][:200]}...\n")


def experiment_alpha():
  """Подпункт 5: Исследование влияния параметра alpha"""
  test_queries = [
      "искусственный интеллект",
      "машинное обучение",
      "нейронные сети",
      "компьютерное зрение",
      "обработка естественного языка",
  ]
  alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]

  with open("corpus.json", "r", encoding="utf-8") as f:
    corpus = {doc["id"]: doc for doc in json.load(f)}
  with open("inverted_index.json", "r", encoding="utf-8") as f:
    bm25_data = json.load(f)
  with open("neural_model.pkl", "rb") as f:
    neural_data = pickle.load(f)
  model = SentenceTransformer(neural_data["model_name"])

  print(
      "\n"
      + "=" * 80
      + "\nЭКСПЕРИМЕНТ: Влияние параметра alpha на результаты выдачи\n"
      + "=" * 80
  )

  for q in test_queries:
    print(f"\nЗапрос: '{q}'")
    candidates = get_candidate_documents(
        q, bm25_data, neural_data, model, corpus, top_n=20
    )
    candidates = normalize_scores(candidates)

    for a in alpha_values:
      for c in candidates:
        c["final_score"] = calculate_final_score(
            c["norm_bm25"], c["norm_cosine"], a
        )

      sorted_c = sorted(candidates, key=lambda x: x["final_score"], reverse=True)
      top_titles = [corpus[c["doc_id"]]["title"] for c in sorted_c[:3]]
      print(f"  alpha = {a:<4} -> Top-3: {top_titles}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Гибридный поиск с нормализацией Min-Max и параметром alpha"
  )
  parser.add_argument(
      "--mode", choices=["search", "experiment"], default="search"
  )
  parser.add_argument("--query", type=str, help="Поисковый запрос")
  parser.add_argument(
      "--alpha",
      type=float,
      default=0.5,
      help="Параметр alpha (от 0.0 до 1.0, по умолчанию 0.5)",
  )
  args = parser.parse_args()

  if args.mode == "experiment":
    experiment_alpha()
  else:
    if not args.query:
      print("Укажите запрос через --query 'ваш запрос'")
    else:
      search_hybrid(args.query, alpha=args.alpha)