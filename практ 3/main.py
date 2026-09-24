import hashlib
from pathlib import Path
import re
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pypdf import PdfReader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from unidecode import unidecode

# ==========================================
# 1. НАСТРОЙКА ПУТЕЙ И ТЕМ
# ==========================================
DATA_DIR = Path("data")  # Путь к папочке с данными

# Имена твоих подпапок и их метки
TOPICS = [
    ("topic_a", "A"),  # Например: первая папка (Компьютерное зрение)
    ("topic_b", "B"),  # Например: вторая папка (NLP)
]


# ==========================================
# 2. ФУНКЦИИ ИЗВЛЕЧЕНИЯ И ОЧИСТКИ ТЕКСТА
# ==========================================
def read_pdf_text(path: Path) -> str:
  """Извлечение текста из PDF."""
  try:
    txt = pdfminer_extract_text(str(path))
    if txt and len(txt.strip()) > 100:
      return txt
  except Exception:
    pass
  try:
    reader = PdfReader(str(path))
    pages = [p.extract_text() or '' for p in reader.pages]
    return '\n'.join(pages)
  except Exception as e:
    warnings.warn(f'PDF read failed: {path.name} ({e})')
    return ''


def cut_references(text: str) -> str:
  """Удаление списка литературы."""
  pattern = (
      r'(?:\n|\r|\r\n)(references|литература|список литературы|appendix)\b.*$'
  )
  return re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)


def extract_title_abstract(text: str) -> tuple[str, str]:
  """Выделение заголовка и аннотации."""
  lines = [l.strip() for l in text.splitlines() if l.strip()]
  title = lines[0][:300] if lines else ''
  m = re.search(
      r'(abstract|аннотация)\s*[:\-\–]?\s*(.+?)(\n[A-Z][A-Za-z ]{3,}|$)',
      text,
      flags=re.IGNORECASE | re.DOTALL,
  )
  abstract = (m.group(2).strip() if m else '')[:3000]
  return title, abstract


def hash_text(s: str) -> str:
  """Хеширование для удаления дубликатов."""
  return hashlib.md5(s.encode('utf-8')).hexdigest()


# ==========================================
# 3. ЧТЕНИЕ И СБОР ВСЕХ СТАТЕЙ
# ==========================================
print('--- Загрузка и парсинг PDF файлов ---')
rows = []
for folder, lbl in TOPICS:
  folder_path = DATA_DIR / folder
  pdf_files = sorted([f for f in folder_path.iterdir() if f.suffix.lower() == '.pdf'])
  print(f"Найдено файлов в {folder}: {len(pdf_files)}")

  for pdf in pdf_files:
    raw = read_pdf_text(pdf)
    raw = unidecode(raw)  # Нормализация символов
    raw = re.sub(r'\s+', ' ', raw)

    if len(raw) < 500:  # Пропуск поврежденных/коротких файлов
      continue

    raw = cut_references(raw)
    title, abstract = extract_title_abstract(raw)
    keep = title + '\n' + abstract + '\n' + raw[:15000]  # Ограничение длины

    rows.append({
        'id': pdf.stem,
        'topic': lbl,
        'title': title,
        'abstract': abstract,
        'text': keep,
        'n_chars': len(keep),
    })

df = pd.DataFrame(rows)

if len(df) == 0:
  raise ValueError(
      'Не найдено ни одного корректного PDF файла! Проверь пути к папкам.'
  )

# Удаление возможных дубликатов
df['hash'] = df['text'].apply(hash_text)
df = df.drop_duplicates(subset='hash').drop(columns=['hash']).reset_index(
    drop=True
)

print(
    f'Успешно загружено документов: {len(df)} | Тема A: {sum(df.topic=="A")} |'
    f' Тема B: {sum(df.topic=="B")}\n'
)

# ==========================================
# 4. ПРЕДОБРАБОТКА ТЕКСТА И ТОКЕНИЗАЦИЯ
# ==========================================
RU_STOP = {
    'это',
    'также',
    'который',
    'которая',
    'которые',
    'данный',
    'далее',
    'например',
    'таким',
    'образом',
    'рисунок',
    'таблица',
    'метод',
    'работа',
    'результат',
    'вывод',
    'согласно',
    'проведен',
    'представлен',
}
EN_STOP = {
    'the',
    'and',
    'for',
    'with',
    'from',
    'that',
    'this',
    'these',
    'those',
    'into',
    'based',
    'using',
    'results',
    'conclusion',
    'figure',
    'table',
    'method',
    'study',
    'paper',
    'approach',
    'analysis',
}
COMMON_STOP = RU_STOP | EN_STOP | {'et', 'al', 'eg', 'ie'}

REF_PAT = re.compile(r'\[\d{1,3}\]|\([^\)]*\d{4}[^\)]*\)')
URL_PAT = re.compile(r'https?://\S+|doi:\S+', re.IGNORECASE)


def simple_tokenize(text: str) -> list[str]:
  text = text.lower()
  text = URL_PAT.sub(' ', text)
  text = REF_PAT.sub(' ', text)
  text = re.sub(r'[^a-zа-яё0-9\- ]', ' ', text)
  text = re.sub(r'\d+[\-–]\d+', ' ', text)
  toks = [t for t in text.split() if len(t) > 2 and t not in COMMON_STOP]
  toks = [t for t in toks if not t.startswith('-') and not t.endswith('-')]
  return toks


def preprocess(text: str) -> str:
  return ' '.join(simple_tokenize(text))


df['text_clean'] = df['text'].apply(preprocess)

# ==========================================
# 5. ВЕКТОРИЗАЦИЯ (TF-IDF + SVD)
# ==========================================
print('--- Векторизация текстов ---')
# Настроено под небольшое количество документов (20 шт)
tfidf = TfidfVectorizer(
    min_df=2, max_df=0.85, ngram_range=(1, 2), max_features=10_000
)
X = tfidf.fit_transform(df['text_clean'])

use_svd = True
if use_svd:
  # Количество компонентов ограничено количеством документов
  n_comp = min(10, X.shape[0] - 1, X.shape[1] - 1)
  svd = TruncatedSVD(n_components=n_comp, random_state=42)
  lsa = make_pipeline(svd, Normalizer(copy=False))
  Xm = lsa.fit_transform(X)
else:
  Xm = X

# ==========================================
# 6. КЛАСТЕРИЗАЦИЯ K-MEANS И ПОДБОР K
# ==========================================
print('--- Обучение K-Means ---')


def scan_k(Xm, ks=range(2, min(6, len(df)))):
  inertias, sils = [], []
  for k in ks:
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    labels = km.fit_predict(Xm)
    inertias.append(km.inertia_)
    sils.append(
        silhouette_score(Xm, labels, sample_size=min(10000, Xm.shape[0]))
    )
  return list(ks), inertias, sils


ks, inertias, sils = scan_k(Xm)

# График локтя и силуэта
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(ks, inertias, marker='o')
plt.title('Elbow (inertia)')
plt.xlabel('k')
plt.ylabel('inertia')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(ks, sils, marker='o', color='orange')
plt.title('Silhouette vs k')
plt.xlabel('k')
plt.ylabel('silhouette')
plt.grid(True)
plt.show()

# Базовый выбор k=2 (так как у нас 2 темы)
k = 2
kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
labels_pred = kmeans.fit_predict(Xm)
df['cluster'] = labels_pred

# ==========================================
# 7. ИНТЕРПРЕТАЦИЯ КЛАСТЕРОВ (ТОП-ТЕРМИНЫ)
# ==========================================
print('\n==========================================')
print('   ТОП-20 ТЕРМИНОВ ПО КЛАСТЕРАМ')
print('==========================================')


def top_terms_per_cluster(tfidf, labels, X_sparse, topn=20):
  terms = np.array(tfidf.get_feature_names_out())
  out = {}
  for c in range(labels.max() + 1):
    rows = np.where(labels == c)[0]
    if len(rows) == 0:
      out[c] = []
      continue
    mean_vec = X_sparse[rows].mean(axis=0).A1
    idx = np.argsort(mean_vec)[::-1][:topn]
    out[c] = terms[idx].tolist()
  return out


top_terms = top_terms_per_cluster(tfidf, labels_pred, X, topn=20)
for c, terms in top_terms.items():
  print(f"\nCluster {c}: {', '.join(terms)}")

print('\n==========================================')
print('   ПРИМЕРЫ СТАТЕЙ В КЛАСТЕРАХ')
print('==========================================')
for c in range(k):
  ex = df[df['cluster'] == c].head(5)
  print(f'\n=== Cluster {c} (всего файлов: {sum(df["cluster"] == c)}) ===')
  for _, r in ex.iterrows():
    print(f"[{r['id']}] Исходная тема: {r['topic']} | Заголовок: {r['title'][:100]}")

# ==========================================
# 8. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ (PCA 2D)
# ==========================================
pca2 = PCA(n_components=2, random_state=42).fit_transform(
    Xm if use_svd else X.toarray()
)

plt.figure(figsize=(9, 6))
scatter = plt.scatter(
    pca2[:, 0],
    pca2[:, 1],
    c=labels_pred,
    cmap='Set1',
    s=100,
    alpha=0.8,
    edgecolors='k',
)
plt.title(f'K-means Clustering, k={k} (PCA 2D)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)

# Подписи названий файлов на графике
for i, txt in enumerate(df['id']):
  plt.annotate(
      txt[:10],
      (pca2[i, 0], pca2[i, 1]),
      fontsize=8,
      xytext=(5, 5),
      textcoords='offset points',
  )

plt.colorbar(scatter, label='Номер кластера')
plt.show()