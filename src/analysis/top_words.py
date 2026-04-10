"""
상위 20 단어 빈도 분석 스크립트
사용법:
  uv run src/analysis/top_words.py --platform youtube
  uv run src/analysis/top_words.py --platform reddit
  uv run src/analysis/top_words.py --platform youtube --top-n 30
"""

import argparse
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# ── 경로 설정 ────────────────────────────────────────────────────────────────

PLATFORM_PATHS = {
    "youtube": "data/processed/youtube_cleaned.csv",
    "reddit":  "data/processed/reddit_cleaned.csv",
}
OUTPUT_DIR = "results/figures"
RESULTS_DIR = "results/tables"

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


def load_data(platform: str) -> pd.DataFrame:
    path = PLATFORM_PATHS[platform]
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일 없음: {path}")
    df = pd.read_csv(path, dtype=str)
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]
    print(f"로드 완료 — {len(df):,}개 댓글")
    return df


def tokenize(df: pd.DataFrame) -> list[str]:
    """소문자 알파벳 토큰만 추출, 불용어 제거, 2글자 이상"""
    tokens = []
    for text in df["text"]:
        words = re.findall(r"[a-z]{2,}", str(text).lower())
        tokens.extend(w for w in words if w not in STOPWORDS)
    return tokens


def plot_top_words(counter: Counter, top_n: int, platform: str):
    words, counts = zip(*counter.most_common(top_n))

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(list(reversed(words)), list(reversed(counts)), color="steelblue")

    # 막대 끝에 숫자 표시
    for bar, count in zip(bars, reversed(counts)):
        ax.text(
            bar.get_width() + max(counts) * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}",
            va="center", ha="left", fontsize=9,
        )

    ax.set_xlabel("빈도", fontsize=12)
    ax.set_title(f"Top {top_n} Words — {platform.capitalize()} Comments", fontsize=14)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.margins(x=0.12)
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"top_words_{platform}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"그래프 저장 → {out_path}")


def save_csv(counter: Counter, top_n: int, platform: str):
    rows = [{"rank": i+1, "word": w, "count": c}
            for i, (w, c) in enumerate(counter.most_common(top_n))]
    df_out = pd.DataFrame(rows)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"top_words_{platform}.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"CSV 저장   → {out_path}")

    # 콘솔 출력
    print(f"\n{'순위':>4}  {'단어':<20} {'빈도':>8}")
    print("-" * 36)
    for _, row in df_out.iterrows():
        print(f"{int(row['rank']):>4}  {row['word']:<20} {int(row['count']):>8,}")


def main():
    parser = argparse.ArgumentParser(description="상위 단어 빈도 분석")
    parser.add_argument("--platform", choices=["youtube", "reddit"], required=True)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    print(f"[상위 단어] 플랫폼: {args.platform}")
    df = load_data(args.platform)
    tokens = tokenize(df)
    print(f"전체 토큰 수: {len(tokens):,}개 / 고유 단어: {len(set(tokens)):,}개")

    counter = Counter(tokens)
    plot_top_words(counter, args.top_n, args.platform)
    save_csv(counter, args.top_n, args.platform)


if __name__ == "__main__":
    main()