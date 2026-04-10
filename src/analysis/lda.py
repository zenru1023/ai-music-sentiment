import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from gensim import corpora, models
from gensim.models import CoherenceModel

# ── 경로 설정 ────────────────────────────────────────────────────────────────

PLATFORM_PATHS = {
    "youtube": "data/processed/youtube_cleaned.csv",
    "reddit":  "data/processed/reddit_cleaned.csv",
}
OUTPUT_MODELS  = "results/models"
OUTPUT_TABLES  = "results/tables"
OUTPUT_FIGURES = "results/figures"

# ── 불용어 ───────────────────────────────────────────────────────────────────

STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "of", "to", "and", "or", "but",
    "this", "that", "be", "are", "was", "were", "for", "with", "as", "at",
    "by", "from", "on", "have", "has", "had", "not", "no", "so", "if",
    "do", "does", "did", "will", "can", "just", "know", "think", "really",
    "like", "get", "got", "make", "made", "way", "even", "thing", "things",
    "people", "one", "still", "much", "well", "also", "would", "could",
    "want", "use", "used", "using", "time", "back", "going", "go", "see",
    "lot", "something", "actually", "i", "you", "he", "she", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "its", "our",
    "their", "what", "when", "where", "who", "how", "why", "more", "than",
    "then", "there", "here", "about", "up", "out", "into", "over", "after",
    "good", "great", "very", "too", "all", "some", "any", "new",
    "music", "song", "ai", "comment", "video", "channel", "post",
}


# ── 데이터 로드 및 토큰화 ────────────────────────────────────────────────────

def load_and_tokenize(platform: str) -> tuple[pd.DataFrame, list[list[str]]]:
    path = PLATFORM_PATHS[platform]
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일 없음: {path}")
    df = pd.read_csv(path, dtype=str)
    df = df[df["text"].notna() & (df["text"].str.strip() != "")].copy()
    print(f"로드 완료 — {len(df):,}개 댓글")

    def tokenize(text: str) -> list[str]:
        words = re.findall(r"[a-z]{3,}", str(text).lower())
        return [w for w in words if w not in STOPWORDS]

    tokenized = df["text"].apply(tokenize).tolist()
    # 빈 문서 제거
    valid = [(i, t) for i, t in enumerate(tokenized) if len(t) >= 3]
    valid_idx, tokenized = zip(*valid) if valid else ([], [])
    df = df.iloc[list(valid_idx)].reset_index(drop=True)

    print(f"유효 문서: {len(tokenized):,}개 (토큰 3개 미만 제외)")
    return df, list(tokenized)


# ── LDA 학습 ─────────────────────────────────────────────────────────────────

def train_lda(
    tokenized: list[list[str]],
    num_topics: int,
    passes: int,
    workers: int,
) -> tuple[corpora.Dictionary, list, models.LdaMulticore]:
    print(f"\n[LDA] 학습 시작 — 토픽 수: {num_topics}, passes: {passes}")

    dictionary = corpora.Dictionary(tokenized)
    # 너무 희귀하거나 너무 흔한 단어 제거
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized]

    lda = models.LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        workers=workers,
        random_state=42,
        per_word_topics=True,
    )

    # Coherence Score (c_v)
    coherence_model = CoherenceModel(
        model=lda, texts=tokenized, dictionary=dictionary, coherence="c_v"
    )
    score = coherence_model.get_coherence()
    print(f"Coherence Score (c_v): {score:.4f}")

    return dictionary, corpus, lda


# ── 시각화 ───────────────────────────────────────────────────────────────────

def plot_topics(lda: models.LdaMulticore, num_topics: int, platform: str, top_n: int = 10):
    cols = min(3, num_topics)
    rows = (num_topics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten() if num_topics > 1 else [axes]

    for topic_idx in range(num_topics):
        top_words = lda.show_topic(topic_idx, topn=top_n)
        words, weights = zip(*top_words)
        ax = axes[topic_idx]
        ax.barh(list(reversed(words)), list(reversed(weights)), color="steelblue")
        ax.set_title(f"Topic {topic_idx + 1}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Weight")
        ax.tick_params(axis="y", labelsize=9)

    # 남는 subplot 숨기기
    for i in range(num_topics, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f"LDA Topics — {platform.capitalize()} Comments", fontsize=14, y=1.01)
    plt.tight_layout()

    os.makedirs(OUTPUT_FIGURES, exist_ok=True)
    out_path = os.path.join(OUTPUT_FIGURES, f"lda_{platform}_topics.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"그래프 저장 → {out_path}")


# ── 저장 ─────────────────────────────────────────────────────────────────────

def save_results(
    df: pd.DataFrame,
    corpus: list,
    lda: models.LdaMulticore,
    dictionary: corpora.Dictionary,
    platform: str,
    num_topics: int,
):
    os.makedirs(OUTPUT_TABLES, exist_ok=True)
    os.makedirs(OUTPUT_MODELS, exist_ok=True)

    # 토픽별 상위 단어
    topic_rows = []
    for topic_idx in range(num_topics):
        top_words = lda.show_topic(topic_idx, topn=15)
        for rank, (word, weight) in enumerate(top_words, 1):
            topic_rows.append({
                "topic": topic_idx + 1,
                "rank": rank,
                "word": word,
                "weight": round(weight, 6),
            })
    topics_df = pd.DataFrame(topic_rows)
    topics_path = os.path.join(OUTPUT_TABLES, f"lda_{platform}_topics.csv")
    topics_df.to_csv(topics_path, index=False, encoding="utf-8-sig")
    print(f"토픽 단어 저장 → {topics_path}")

    # 댓글별 주요 토픽
    doc_topics = []
    for i, bow in enumerate(corpus):
        # Explicitly cast to int and float to satisfy the type checker
        raw_topics = lda.get_document_topics(bow)
        topic_probs = {int(k): float(v) for k, v in raw_topics} # type: ignore
        
        if topic_probs:
            dominant = max(topic_probs, key=lambda k: topic_probs[k]) + 1
            dominant_prob = max(topic_probs.values())
        else:
            dominant = -1
            dominant_prob = 0.0

        doc_topics.append({
            "dominant_topic": dominant, 
            "topic_prob": round(float(dominant_prob), 4)
        })

    doc_df = pd.DataFrame(doc_topics)
    result_df = pd.concat([df.reset_index(drop=True), doc_df], axis=1)
    docs_path = os.path.join(OUTPUT_TABLES, f"lda_{platform}_docs.csv")
    result_df.to_csv(docs_path, index=False, encoding="utf-8-sig")
    print(f"문서별 토픽 저장 → {docs_path}")

    # 모델 저장
    model_path = os.path.join(OUTPUT_MODELS, f"lda_{platform}.model")
    lda.save(model_path)
    dictionary.save(os.path.join(OUTPUT_MODELS, f"lda_{platform}.dict"))
    print(f"모델 저장 → {model_path}")


# ── 콘솔 출력 ────────────────────────────────────────────────────────────────

def print_topics(lda: models.LdaMulticore, num_topics: int):
    print("\n" + "=" * 50)
    print("토픽 요약")
    print("=" * 50)
    for i in range(num_topics):
        top_words = [w for w, _ in lda.show_topic(i, topn=10)]
        print(f"Topic {i+1:>2}: {', '.join(top_words)}")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LDA 토픽 모델링")
    parser.add_argument("--platform", choices=["youtube", "reddit"], required=True)
    parser.add_argument("--num-topics", type=int, default=7,
                        help="토픽 수 (기본값: 7)")
    parser.add_argument("--passes", type=int, default=15,
                        help="학습 반복 횟수 (기본값: 15)")
    parser.add_argument("--workers", type=int, default=2,
                        help="멀티코어 워커 수 (기본값: 2)")
    args = parser.parse_args()

    print(f"[LDA] 플랫폼: {args.platform}")
    df, tokenized = load_and_tokenize(args.platform)
    dictionary, corpus, lda = train_lda(
        tokenized, args.num_topics, args.passes, args.workers
    )
    print_topics(lda, args.num_topics)
    plot_topics(lda, args.num_topics, args.platform)
    save_results(df, corpus, lda, dictionary, args.platform, args.num_topics)
    print("\n완료.")


if __name__ == "__main__":
    main()