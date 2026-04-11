import os
import pandas as pd
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim_models

# ── 설정 ───────────────────────────────

PLATFORM = "youtube"

MODEL_PATH = f"results/models/lda/lda_{PLATFORM}.model"
DICT_PATH  = f"results/models/lda/lda_{PLATFORM}.dict"

OUTPUT_TABLES = "results/tables/lda"
OUTPUT_FIGURES = "results/figures/lda"

os.makedirs(OUTPUT_TABLES, exist_ok=True)
os.makedirs(OUTPUT_FIGURES, exist_ok=True)

# ── 데이터 (pyLDAvis용 corpus 필요) ──
# ⚠️ 중요: LDA 돌렸던 동일 데이터 필요
DATA_PATH = f"data/processed/topic/{PLATFORM}.csv"

# ── 로드 ───────────────────────────────

print("[LOAD] model & dict")

lda = models.LdaModel.load(MODEL_PATH)
dictionary = corpora.Dictionary.load(DICT_PATH)

df = pd.read_csv(DATA_PATH, dtype=str)
df = df[df["text"].notna() & (df["text"].str.strip() != "")].copy()

import re

def tokenize(text):
    return re.findall(r"[a-z]{3,}", str(text).lower())

tokenized = df["text"].apply(tokenize).tolist()
tokenized = [t for t in tokenized if len(t) >= 3]

corpus = [dictionary.doc2bow(t) for t in tokenized]


# ── 1️⃣ 토픽 단어 저장 ─────────────────

topic_rows = []

for topic_id in range(lda.num_topics):
    for rank, (word, weight) in enumerate(lda.show_topic(topic_id, topn=15), 1):
        topic_rows.append({
            "topic": topic_id + 1,
            "rank": rank,
            "word": word,
            "weight": round(weight, 6)
        })

topics_df = pd.DataFrame(topic_rows)

topics_path = os.path.join(OUTPUT_TABLES, f"{PLATFORM}_topics.csv")
topics_df.to_csv(topics_path, index=False, encoding="utf-8-sig")

print(f"[SAVE] topics → {topics_path}")


# ── 2️⃣ 문서별 토픽 ───────────────────

doc_rows = []

for bow in corpus:
    topics = lda.get_document_topics(bow)

    if topics:
        dominant = max(topics, key=lambda x: x[1])
        doc_rows.append({
            "dominant_topic": dominant[0] + 1,
            "prob": round(dominant[1], 4)
        })
    else:
        doc_rows.append({
            "dominant_topic": -1,
            "prob": 0.0
        })

doc_df = pd.DataFrame(doc_rows)

final_df = pd.concat([df.reset_index(drop=True), doc_df], axis=1)

doc_path = os.path.join(OUTPUT_TABLES, f"{PLATFORM}_doc_topics.csv")
final_df.to_csv(doc_path, index=False, encoding="utf-8-sig")

print(f"[SAVE] doc topics → {doc_path}")


# ── 3️⃣ pyLDAvis HTML ─────────────────

print("[VIS] generating HTML...")

vis = pyLDAvis.gensim_models.prepare(
    lda,
    corpus,
    dictionary,
    sort_topics=False
)

html_path = os.path.join(OUTPUT_FIGURES, f"{PLATFORM}_ldavis.html")
pyLDAvis.save_html(vis, html_path)

print(f"[SAVE] HTML → {html_path}")


print("\nDONE ✔")