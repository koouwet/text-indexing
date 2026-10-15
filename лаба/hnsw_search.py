import argparse
import json
import pickle
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "cointegrated/rubert-tiny2"


def build_hnsw_index():
  print("Загрузка нейросетевых векторов из Части IV...")
  with open("neural_model.pkl", "rb") as f:
    data = pickle.load(f)

  # Faiss требует тип float32 и нормированные векторы для косинусного расстояния через IndexHNSWFlat
  doc_vectors = data["doc_vectors"].astype(np.float32)
  faiss.normalize_L2(doc_vectors)
  doc_ids = data["doc_ids"]
  num_elements, dim = doc_vectors.shape

  # M — количество связей на узел (обычно 16-32)
  M = 16
  print(
      f"Построение HNSW индекса через Faiss (элементов: {num_elements},"
      f" размерность: {dim})..."
  )

  # Создаем индекс на основе Inner Product (для нормированных векторов это эквивалентно косинусному сходству)
  index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
  index.hnsw.efConstruction = 200

  start_time = time.time()
  index.add(doc_vectors)
  build_time = time.time() - start_time

  # Установка efSearch по умолчанию
  index.hnsw.efSearch = 50

  # Сохранение индекса и маппинга ID
  faiss.write_index(index, "faiss_hnsw.index")
  with open("hnsw_meta.pkl", "wb") as f:
    pickle.dump({"doc_ids": doc_ids}, f)

  print(f"Индекс успешно построен и сохранен!")
  print(f"Время построения: {build_time:.4f} секунд")
  print(f"Параметры: M={M}, efConstruction=200")


def search_hnsw(query, ef_search=50, top_k=5):
  with open("hnsw_meta.pkl", "rb") as f:
    meta = pickle.load(f)
  doc_ids = meta["doc_ids"]

  with open("neural_model.pkl", "rb") as f:
    neural_data = pickle.load(f)

  model = SentenceTransformer(neural_data["model_name"])

  with open("corpus.json", "r", encoding="utf-8") as f:
    corpus = {doc["id"]: doc for doc in json.load(f)}

  # Преобразование запроса в вектор (измеряем отдельно)
  q_start = time.time()
  query_vector = model.encode([query]).astype(np.float32)
  faiss.normalize_L2(query_vector)
  vectorize_time = time.time() - q_start

  # Загрузка индекса
  index = faiss.read_index("faiss_hnsw.index")
  index.hnsw.efSearch = ef_search

  # Поиск по индексу
  s_start = time.time()
  distances, labels = index.search(query_vector, top_k)
  search_time = time.time() - s_start

  print(f"\nРезультаты HNSW поиска (efSearch={ef_search}) для: '{query}'")
  print(
      f"Время векторизации: {vectorize_time*1000:.2f} мс | Время поиска в индексе:"
      f" {search_time*1000:.2f} мс"
  )

  results = []
  for i in range(top_k):
    idx = labels[0][i]
    score = distances[0][i]  # Скалярное произведение нормированных векторов
    doc_id = doc_ids[idx]
    doc = corpus[doc_id]
    results.append(doc_id)
    print(f"{i+1}. [{doc_id}] {doc['title']} (Score: {score:.4f})")
    print(f"   {doc.get('summary', doc['text'][:150])}\n")

  return results


def evaluate_ef_search():
  """Эксперимент для пункта 5: исследование влияния efSearch"""
  test_queries = [
      "искусственный интеллект",
      "машинное обучение",
      "нейронные сети",
      "глубокое обучение",
      "обработка естественного языка",
  ]
  ef_values = [10, 25, 50, 100]

  with open("neural_model.pkl", "rb") as f:
    neural_data = pickle.load(f)
  doc_vectors = neural_data["doc_vectors"].astype(np.float32)
  faiss.normalize_L2(doc_vectors)
  doc_ids = neural_data["doc_ids"]
  model = SentenceTransformer(neural_data["model_name"])

  index = faiss.read_index("faiss_hnsw.index")

  print(
      "\n"
      + "=" * 60
      + "\nЭКСПЕРИМЕНТ: Оценка точности и скорости для разных efSearch\n"
      + "=" * 60
  )
  print(
      f"{'efSearch':<10} | {'Ср. доля совпадений':<20} | {'Ср. время поиска (мс)':<22}"
  )
  print("-" * 60)

  for ef in ef_values:
    index.hnsw.efSearch = ef
    match_ratios = []
    search_times = []

    for q in test_queries:
      q_vec = model.encode([q]).astype(np.float32)
      faiss.normalize_L2(q_vec)

      # Точный поиск (эталон через матричное умножение)
      exact_sims = np.dot(doc_vectors, q_vec[0])
      exact_top5 = set(
          [doc_ids[idx] for idx in np.argsort(exact_sims)[::-1][:5]]
      )

      # Приближенный поиск с замерением времени
      t0 = time.time()
      _, labels = index.search(q_vec, 5)
      t_elapsed = time.time() - t0
      search_times.append(t_elapsed * 1000)

      approx_top5 = set([doc_ids[idx] for idx in labels[0]])

      # Доля совпадений = количество общих документов / 5
      common = len(exact_top5.intersection(approx_top5))
      match_ratios.append(common / 5.0)

    avg_match = np.mean(match_ratios)
    avg_time = np.mean(search_times)
    print(f"{ef:<10} | {avg_match:<20.2f} | {avg_time:<22.4f}")
  print("=" * 60)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="HNSW Векторное индексирование (Faiss)")
  parser.add_argument(
      "--mode",
      choices=["build", "search", "evaluate"],
      required=True,
      help="Режим работы",
  )
  parser.add_argument("--query", type=str, help="Поисковый запрос")
  parser.add_argument(
      "--ef", type=int, default=50, help="Параметр efSearch (по умолчанию 50)"
  )
  args = parser.parse_args()

  if args.mode == "build":
    build_hnsw_index()
  elif args.mode == "search":
    if not args.query:
      print("Укажите поисковый запрос с помощью --query 'ваш запрос'")
    else:
      search_hnsw(args.query, ef_search=args.ef)
  elif args.mode == "evaluate":
    evaluate_ef_search()