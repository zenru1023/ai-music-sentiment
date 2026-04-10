import argparse
import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

# ── 경로 설정 ────────────────────────────────────────────────────────────────

PLATFORM_PATHS = {
    "youtube": "data/processed/youtube_cleaned.csv",
    "reddit":  "data/processed/reddit_cleaned.csv",
}
OUTPUT_MODELS  = "results/models"
OUTPUT_TABLES  = "results/tables"
OUTPUT_FIGURES = "results/figures"

# ── 분석 대상 핵심 단어 ──────────────────────────────────────────────────────
# t-SNE 시각화 및 유사어 분석에 사용
SEED_WORDS = [
    "suno", "udio", "generated", "artificial",
    "creativity", "human", "copyright", "emotion",
    "quality", "replace", "artist", "original",
    "industry", "future", "authentic", "tool",
]

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
    "comment", "video", "channel", "post", "reddit", "youtube",
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

    sentences = df["text"].apply(tokenize).tolist()
    sentences = [s for s in sentences if len(s) >= 2]
    print(f"유효 문장: {len(sentences):,}개")
    return df, sentences


# ── Word2Vec 학습 ─────────────────────────────────────────────────────────────

def train_word2vec(
    sentences: list[list[str]],
    vector_size: int,
    window: int,
    min_count: int,
    workers: int,
) -> Word2Vec:
    print(f"\n[Word2Vec] 학습 중 — vector_size={vector_size}, window={window}")
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,          # Skip-gram (댓글처럼 짧은 텍스트에 유리)
        epochs=10,
        seed=42,
    )
    vocab_size = len(model.wv)
    print(f"어휘 크기: {vocab_size:,}개")
    return model


# ── 유사어 분석 ──────────────────────────────────────────────────────────────

def analyze_similar_words(model: Word2Vec, platform: str, top_n: int = 10):
    rows = []
    print("\n" + "=" * 50)
    print("핵심 단어 유사어 분석")
    print("=" * 50)

    available = [w for w in SEED_WORDS if w in model.wv]
    missing = [w for w in SEED_WORDS if w not in model.wv]
    if missing:
        print(f"[주의] 어휘에 없는 단어 (min_count 미달): {', '.join(missing)}")

    for word in available:
        similars = model.wv.most_similar(word, topn=top_n)
        print(f"\n'{word}' 유사어:")
        for rank, (sim_word, score) in enumerate(similars, 1):
            print(f"  {rank:>2}. {sim_word:<20} {score:.4f}")
            rows.append({
                "seed_word": word,
                "rank": rank,
                "similar_word": sim_word,
                "similarity": round(score, 4),
            })

    os.makedirs(OUTPUT_TABLES, exist_ok=True)
    out_path = os.path.join(OUTPUT_TABLES, f"w2v_{platform}_similar.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n유사어 CSV 저장 → {out_path}")


# ── t-SNE 시각화 ─────────────────────────────────────────────────────────────

def plot_tsne(model: Word2Vec, platform: str, n_words: int = 100):
    """어휘 상위 n_words 단어를 t-SNE로 2D 시각화"""
    vocab = list(model.wv.index_to_key[:n_words])
    vectors = np.array([model.wv[w] for w in vocab])

    perplexity = min(30, len(vocab) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
    coords = tsne.fit_transform(vectors)

    # seed words 강조 표시
    seed_set = set(SEED_WORDS)

    fig, ax = plt.subplots(figsize=(14, 10))
    for i, word in enumerate(vocab):
        is_seed = word in seed_set
        color = "#E53935" if is_seed else "#1565C0"
        size  = 60 if is_seed else 20
        ax.scatter(coords[i, 0], coords[i, 1], c=color, s=size, alpha=0.7)
        fontsize = 10 if is_seed else 7
        weight   = "bold" if is_seed else "normal"
        
        # word를 str()로 감싸서 Pylance에게 문자열임을 확신시켜 줍니다.
        ax.annotate(str(word), (coords[i, 0], coords[i, 1]),
                    fontsize=fontsize, fontweight=weight,
                    xytext=(3, 3), textcoords="offset points", alpha=0.9)

    ax.set_title(f"Word2Vec t-SNE — {platform.capitalize()} (top {n_words} words)",
                 fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

    # 범례
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E53935",
               markersize=10, label="Seed words"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1565C0",
               markersize=7, label="Other words"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    os.makedirs(OUTPUT_FIGURES, exist_ok=True)
    out_path = os.path.join(OUTPUT_FIGURES, f"w2v_{platform}_tsne.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"t-SNE 저장 → {out_path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Word2Vec 분석")
    parser.add_argument("--platform", choices=["youtube", "reddit"], required=True)
    parser.add_argument("--vector-size", type=int, default=150,
                        help="임베딩 차원 수 (기본값: 150)")
    parser.add_argument("--window", type=int, default=5,
                        help="컨텍스트 윈도우 크기 (기본값: 5)")
    parser.add_argument("--min-count", type=int, default=3,
                        help="최소 등장 횟수 (기본값: 3)")
    parser.add_argument("--workers", type=int, default=4,
                        help="병렬 워커 수 (기본값: 4)")
    parser.add_argument("--tsne-words", type=int, default=100,
                        help="t-SNE 시각화할 단어 수 (기본값: 100)")
    args = parser.parse_args()

    print(f"[Word2Vec] 플랫폼: {args.platform}")
    _, sentences = load_and_tokenize(args.platform)

    model = train_word2vec(
        sentences,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
    )

    analyze_similar_words(model, args.platform)
    plot_tsne(model, args.platform, n_words=args.tsne_words)

    # 모델 저장
    os.makedirs(OUTPUT_MODELS, exist_ok=True)
    model_path = os.path.join(OUTPUT_MODELS, f"w2v_{args.platform}.model")
    model.save(model_path)
    print(f"모델 저장 → {model_path}")
    print("\n완료.")


if __name__ == "__main__":
    main()