import argparse
import json
import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import pymorphy3

morph = pymorphy3.MorphAnalyzer()


def preprocess_text(text):
  """Предобработка текста для векторизации"""
  words = re.findall(r"[a-zа-яё]+", text.lower())
  return " ".join([morph.parse(w)[0].normal_form for w in words])


def train_vector_model():
  print("Режим обучения: построение TF-IDF и SVD моделей...")
  with open("corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

  texts = [doc["title"] + " " + doc["text"] for doc in corpus]
  processed_texts = [preprocess_text(t) for t in texts]

  # 1-2. Создание словаря и матрицы TF-IDF
  vectorizer = TfidfVectorizer(max_features=10000)
  tfidf_matrix = vectorizer.fit_transform(processed_texts)

  # Гарантированный вывод заполненных ненулевых ячейки для отчета
  print("\n--- Фрагмент матрицы TF-IDF (значимые веса) ---")
  feature_names = vectorizer.get_feature_names_out()
  dense_matrix = tfidf_matrix.toarray()

  non_zero_rows, non_zero_cols = np.nonzero(dense_matrix)

  if len(non_zero_rows) > 0:
    # Находим колонки с наибольшим количеством заполненных ячеек
    col_counts = np.bincount(
        non_zero_cols, minlength=dense_matrix.shape[1]
    )
    top_cols = np.argsort(col_counts)[::-1][:5]

    # Находим документы, в которых эти слова точно встречаются
    sample_rows = []
    for r in np.unique(non_zero_rows):
      if any(dense_matrix[r, c] > 0 for c in top_cols):
        sample_rows.append(r)
      if len(sample_rows) >= 5:
        break
    if not sample_rows:
      sample_rows = np.unique(non_zero_rows)[:5]

    print(f"{'Документ':<10}", end="")
    for col_idx in top_cols:
      print(f"{feature_names[col_idx][:10]:<12}", end="")
    print()

    for row_idx in sample_rows:
      print(f"doc_{row_idx+1:<6}", end="")
      for col_idx in top_cols:
        val = dense_matrix[row_idx, col_idx]
        print(f"{val:<12.4f}", end="")
      print()
  print("-" * 70 + "\n")

  # 5. Усеченное сингулярное разложение (SVD) для уменьшения размерности
  n_components = min(100, tfidf_matrix.shape[0] - 1)
  svd = TruncatedSVD(n_components=n_components, random_state=42)
  reduced_matrix = svd.fit_transform(tfidf_matrix)

  # Сохранение моделей и векторов
  data = {"doc_vectors": reduced_matrix, "doc_ids": [doc["id"] for doc in corpus]}

  with open("vector_model.pkl", "wb") as f:
    pickle.dump({"vectorizer": vectorizer, "svd": svd, "data": data}, f)

  print(
      f"Векторная модель успешно обучена! Итоговая размерность векторов:"
      f" {reduced_matrix.shape[1]}"
  )


def search_vector(query, top_k=5):
  with open("vector_model.pkl", "rb") as f:
    model_pack = pickle.load(f)

  vectorizer = model_pack["vectorizer"]
  svd = model_pack["svd"]
  data = model_pack["data"]
  doc_vectors = data["doc_vectors"]
  doc_ids = data["doc_ids"]

  with open("corpus.json", "r", encoding="utf-8") as f:
    corpus = {doc["id"]: doc for doc in json.load(f)}

  # 3. Преобразование запроса в TF-IDF и SVD пространство
  proc_query = preprocess_text(query)
  query_tfidf = vectorizer.transform([proc_query])
  query_vector = svd.transform(query_tfidf)

  # 4. Вычисление косинусного сходства
  similarities = cosine_similarity(query_vector, doc_vectors)[0]

  # Сортировка по убыванию сходства
  sorted_indices = np.argsort(similarities)[::-1]

  print(f"\nРезультаты векторного поиска по запросу: '{query}'")
  results = []
  for i in range(min(top_k, len(sorted_indices))):
    idx = sorted_indices[i]
    doc_id = doc_ids[idx]
    score = similarities[idx]
    doc = corpus[doc_id]
    results.append(doc_id)
    print(f"{i+1}. [{doc_id}] {doc['title']} (Косинусное сходство: {score:.4f})")
    print(f"   {doc['summary']}\n")

  return results


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Векторный поиск TF-IDF + SVD")
  parser.add_argument(
      "--mode", choices=["train", "search"], required=True, help="Режим работы"
  )
  parser.add_argument("--query", type=str, help="Поисковый запрос")
  args = parser.parse_args()

  if args.mode == "train":
    train_vector_model()
  elif args.mode == "search":
    if not args.query:
      print("Укажите поисковый запрос с помощью --query 'ваш запрос'")
    else:
      search_vector(args.query)