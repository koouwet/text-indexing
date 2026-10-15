import argparse
import json
import math
import pickle
import re
import time
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pymorphy3

morph = pymorphy3.MorphAnalyzer()


def preprocess_text(text):
  words = re.findall(r"[a-zа-яё]+", text.lower())
  return [morph.parse(w)[0].normal_form for w in words]


def extract_features(query, doc_id, corpus, bm25_data, vector_pack, neural_pack, model):
  """Шаг 2: Вычисление признаков для пары «запрос — документ»"""
  doc = corpus[doc_id]
  q_tokens = preprocess_text(query)
  doc_title_tokens = set(preprocess_text(doc["title"]))
  doc_text_tokens = preprocess_text(doc["text"])

  # 1. Оценка BM25
  bm25_score = 0.0
  index = bm25_data["index"]
  doc_lengths = bm25_data["doc_lengths"]
  avgdl = bm25_data["avgdl"]
  N = bm25_data["total_docs"]
  k1, b = 1.5, 0.75

  for q in q_tokens:
    if q in index:
      postings = index[q]
      n_q = len(postings)
      idf = max(0, math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0))
      for d_id, freq in postings:
        if d_id == doc_id:
          dl = doc_lengths[doc_id]
          bm25_score += idf * (freq * (k1 + 1)) / (
              freq + k1 * (1 - b + b * (dl / avgdl))
          )

  # 2. Косинусное сходство TF-IDF + SVD
  vectorizer = vector_pack["vectorizer"]
  svd = vector_pack["svd"]
  doc_vectors_svd = vector_pack["data"]["doc_vectors"]
  doc_ids_svd = vector_pack["data"]["doc_ids"]

  q_tfidf = vectorizer.transform([" ".join(q_tokens)])
  q_svd = svd.transform(q_tfidf)
  try:
    idx_svd = doc_ids_svd.index(doc_id)
    tfidf_sim = cosine_similarity(q_svd, [doc_vectors_svd[idx_svd]])[0][0]
  except ValueError:
    tfidf_sim = 0.0

  # 3. Косинусное сходство нейросетевых векторов
  neural_vectors = neural_pack["doc_vectors"]
  neural_ids = neural_pack["doc_ids"]
  q_neural = model.encode([query])
  try:
    idx_neural = neural_ids.index(doc_id)
    neural_sim = cosine_similarity(q_neural, [neural_vectors[idx_neural]])[0][
        0
    ]
  except ValueError:
    neural_sim = 0.0

  # 4. Доля уникальных слов запроса в тексте документа
  unique_q_words = set(q_tokens)
  if unique_q_words:
    in_text_count = sum(
        1 for w in unique_q_words if w in set(doc_text_tokens)
    )
    frac_text = in_text_count / len(unique_q_words)
  else:
    frac_text = 0.0

  # 5. Доля уникальных слов запроса в заголовке документа
  if unique_q_words:
    in_title_count = sum(1 for w in unique_q_words if w in doc_title_tokens)
    frac_title = in_title_count / len(unique_q_words)
  else:
    frac_title = 0.0

  # 6. Длина документа в токенах
  doc_len = len(doc_text_tokens)

  return [bm25_score, tfidf_sim, neural_sim, frac_text, frac_title, doc_len]


def train_ltr_model():
  print("Обучение модели LTR (Градиентный бустинг)...")
  with open("corpus.json", "r", encoding="utf-8") as f:
    corpus_list = json.load(f)
    corpus = {doc["id"]: doc for doc in corpus_list}

  with open("inverted_index.json", "r", encoding="utf-8") as f:
    bm25_data = json.load(f)

  with open("vector_model.pkl", "rb") as f:
    vector_pack = pickle.load(f)

  with open("neural_model.pkl", "rb") as f:
    neural_pack = pickle.load(f)

  model = SentenceTransformer(neural_pack["model_name"])

  # Шаг 1: Формируем тренировочный набор данных (симуляция разметки для 30 запросов)
  train_queries = [
      "искусственный интеллект",
      "машинное обучение",
      "нейронные сети",
      "глубокое обучение",
      "обработка естественного языка",
      "компьютерное зрение",
      "алгоритм сортировки",
      "база данных",
      "веб приложение",
      "язык программирования python",
      "операционная система",
      "компьютерная сеть",
      "информационная безопасность",
      "объектно ориентированное программирование",
      "теория алгоритмов",
      "искусственные нейронные сети",
      "рекуррентная нейронная сеть",
      "сверточная нейронная сеть",
      "градиентный бустинг",
      "метод опорных векторов",
      "логистическая регрессия",
      "кластеризация данных",
      "метод k средних",
      "анализ данных",
      "большие данные",
      "облачные вычисления",
      "распределенные системы",
      "криптография",
      "блокчейн технологии",
      "интернет вещей",
  ]

  X_train = []
  y_train = []
  dataset_records = []

  for q_id, query in enumerate(train_queries):
    # Получаем кандидатов через гибридный поиск (или берем топ-10 случайных/релевантных)
    # Для простоты симуляции берем топ-10 документов по BM25
    query_tokens = preprocess_text(query)
    scores = {}
    index = bm25_data["index"]
    doc_lengths = bm25_data["doc_lengths"]
    avgdl = bm25_data["avgdl"]
    N = bm25_data["total_docs"]
    for q in query_tokens:
      if q in index:
        for d_id, freq in index[q]:
          dl = doc_lengths[d_id]
          sc = freq / (freq + 1.5 * (1 - 0.75 + 0.75 * (dl / avgdl)))
          scores[d_id] = scores.get(d_id, 0.0) + sc

    candidates = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[
        :10
    ]
    if not candidates:
      candidates = [doc["id"] for doc in corpus_list[:10]]

    for rank_idx, doc_id in enumerate(candidates):
      features = extract_features(
          query, doc_id, corpus, bm25_data, vector_pack, neural_pack, model
      )

      # Условная разметка: 1 – полностью, 2 – частично, 3 – не соответствует
      # Эмулируем экспертную разметку на основе ранга
      relevance = 1 if rank_idx < 2 else (2 if rank_idx < 5 else 3)

      X_train.append(features)
      y_train.append(relevance)

      dataset_records.append({
          "query_id": q_id,
          "query_text": query,
          "doc_id": doc_id,
          "relevance": relevance,
      })

  # Сохраняем датасет в JSON
  with open("ltr_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset_records, f, ensure_ascii=False, indent=2)

  # Шаг 4: Обучение модели градиентного бустинга
  reg = GradientBoostingRegressor(
      n_estimators=100, learning_rate=0.1, random_state=42
  )
  reg.fit(X_train, y_train)

  with open("ltr_model.pkl", "wb") as f:
    pickle.dump(reg, f)

  print("Модель LTR успешно обучена и сохранена в ltr_model.pkl!")


def search_ltr(query, top_k=5):
  """Шаг 5: Переупорядочивание результатов поиска"""
  with open("corpus.json", "r", encoding="utf-8") as f:
    corpus = {doc["id"]: doc for doc in json.load(f)}

  with open("inverted_index.json", "r", encoding="utf-8") as f:
    bm25_data = json.load(f)

  with open("vector_model.pkl", "rb") as f:
    vector_pack = pickle.load(f)

  with open("neural_model.pkl", "rb") as f:
    neural_pack = pickle.load(f)

  with open("ltr_model.pkl", "rb") as f:
    reg = pickle.load(f)

  model = SentenceTransformer(neural_pack["model_name"])

  # Получаем кандидатов (например, топ-20 по гибридному поиску)
  # Для упрощения возьмем кандидатов по BM25
  query_tokens = preprocess_text(query)
  scores = {}
  index = bm25_data["index"]
  doc_lengths = bm25_data["doc_lengths"]
  avgdl = bm25_data["avgdl"]
  N = bm25_data["total_docs"]
  for q in query_tokens:
    if q in index:
      for d_id, freq in index[q]:
        dl = doc_lengths[d_id]
        sc = freq / (freq + 1.5 * (1 - 0.75 + 0.75 * (dl / avgdl)))
        scores[d_id] = scores.get(d_id, 0.0) + sc

  candidates = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:20]

  scored_candidates = []
  for doc_id in candidates:
    features = extract_features(
        query, doc_id, corpus, bm25_data, vector_pack, neural_pack, model
    )
    pred_score = reg.predict([features])[0]
    scored_candidates.append((doc_id, pred_score))

  # Сортируем по предсказанной оценке (меньше предсказанный балл релевантности 1-3 -> лучше)
  scored_candidates.sort(key=lambda x: x[1])

  print(f"\nРезультаты LTR-поиска по запросу: '{query}'")
  results = []
  for i, (doc_id, score) in enumerate(scored_candidates[:top_k], 1):
    doc = corpus[doc_id]
    results.append(doc_id)
    print(f"{i}. [{doc_id}] {doc['title']} (Pred Score: {score:.4f})")
    print(f"   {doc.get('summary', doc['text'][:150])}\n")

  return results


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Интеллектуальное ранжирование LTR")
  parser.add_argument(
      "--mode", choices=["train", "search"], required=True, help="Режим работы"
  )
  parser.add_argument("--query", type=str, help="Поисковый запрос")
  args = parser.parse_args()

  if args.mode == "train":
    train_ltr_model()
  elif args.mode == "search":
    if not args.query:
      print("Укажите поисковый запрос с помощью --query 'ваш запрос'")
    else:
      search_ltr(args.query)