import argparse
import json
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Легкая и эффективная модель для русского языка
MODEL_NAME = "cointegrated/rubert-tiny2"


def train_neural_model():
  print(f"Загрузка модели {MODEL_NAME}...")
  model = SentenceTransformer(MODEL_NAME)

  with open("corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

  # Формируем текст для векторизации (заголовок + содержание)
  texts = [
      (doc["title"] + ". " + (doc.get("summary", "") or doc["text"][:300]))
      for doc in corpus
  ]
  doc_ids = [doc["id"] for doc in corpus]

  print("Генерация плотных векторов для документов...")
  doc_vectors = model.encode(texts, show_progress_bar=True)

  # Сохранение векторов и метаданных
  data = {
      "doc_vectors": np.array(doc_vectors),
      "doc_ids": doc_ids,
      "model_name": MODEL_NAME,
  }

  with open("neural_model.pkl", "wb") as f:
    pickle.dump(data, f)

  print(
      f"Нейросетевая модель обучена! Сохранено векторов:"
      f" {len(doc_vectors)}, размерность: {doc_vectors.shape[1]}"
  )


def search_neural(query, top_k=5):
  with open("neural_model.pkl", "rb") as f:
    data = pickle.load(f)

  model = SentenceTransformer(data["model_name"])
  doc_vectors = data["doc_vectors"]
  doc_ids = data["doc_ids"]

  with open("corpus.json", "r", encoding="utf-8") as f:
    corpus = {doc["id"]: doc for doc in json.load(f)}

  # Преобразование запроса в вектор
  query_vector = model.encode([query])

  # Точный поиск по косинусному сходству со всеми векторами
  similarities = cosine_similarity(query_vector, doc_vectors)[0]
  sorted_indices = np.argsort(similarities)[::-1]

  print(
      f"\nРезультаты нейросетевого поиска (точный поиск) по запросу:"
      f" '{query}'"
  )
  results = []
  for i in range(min(top_k, len(sorted_indices))):
    idx = sorted_indices[i]
    doc_id = doc_ids[idx]
    score = similarities[idx]
    doc = corpus[doc_id]
    results.append(doc_id)
    print(f"{i+1}. [{doc_id}] {doc['title']} (Cosine: {score:.4f})")
    print(f"   {doc.get('summary', doc['text'][:150])}\n")

  return results


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Нейросетевой семантический поиск"
  )
  parser.add_argument(
      "--mode", choices=["train", "search"], required=True, help="Режим работы"
  )
  parser.add_argument("--query", type=str, help="Поисковый запрос")
  args = parser.parse_args()

  if args.mode == "train":
    train_neural_model()
  elif args.mode == "search":
    if not args.query:
      print("Укажите запрос с помощью --query 'ваш запрос'")
    else:
      search_neural(args.query)