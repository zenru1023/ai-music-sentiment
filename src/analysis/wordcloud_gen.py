import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, STOPWORDS

# ── 경로 설정 ────────────────────────────────────────────────────────────────

PLATFORM_PATHS = {
    "youtube": "data/processed/youtube_cleaned.csv",
    "reddit":  "data/processed/reddit_cleaned.csv",
}
OUTPUT_DIR = "results/figures"

# ── 불용어 ───────────────────────────────────────────────────────────────────

CUSTOM_STOPWORDS = STOPWORDS | {
    # 너무 일반적인 단어
    "music", "song", "like", "just", "know", "think", "really",
    "good", "great", "make", "made", "way", "even", "thing", "things",
    "people", "one", "get", "got", "still", "much", "well", "also",
    "would", "could", "want", "use", "used", "using", "time", "year",
    "years", "back", "going", "go", "see", "lot", "something", "actually",
    # 플랫폼 노이즈
    "comment", "video", "channel", "post", "reddit", "youtube",
    "deleted", "removed",
}


def load_data(platform: str) -> pd.DataFrame:
    path = PLATFORM_PATHS[platform]
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일 없음: {path}")
    df = pd.read_csv(path, dtype=str)
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]
    print(f"로드 완료 — {len(df):,}개 댓글")
    return df


def build_corpus(df: pd.DataFrame) -> str:
    """모든 댓글을 하나의 문자열로 합침 (숫자·특수문자 제거)"""
    text = " ".join(df["text"].tolist())
    text = re.sub(r"[^a-z\s]", " ", text)   # 소문자·공백만 유지
    text = re.sub(r"\s+", " ", text).strip()
    return text


def generate_wordcloud(corpus: str, platform: str, max_words: int):
    wc = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        stopwords=CUSTOM_STOPWORDS,
        max_words=max_words,
        collocations=False,        # 단어 쌍 비활성화 (단일 토큰 기준)
        colormap="viridis",
        prefer_horizontal=0.85,
    ).generate(corpus)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        f"Word Cloud — {platform.capitalize()} Comments",
        fontsize=20, pad=16,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"wordcloud_{platform}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"저장 완료 → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="워드클라우드 생성")
    parser.add_argument("--platform", choices=["youtube", "reddit"], required=True)
    parser.add_argument("--max-words", type=int, default=150)
    args = parser.parse_args()

    print(f"[워드클라우드] 플랫폼: {args.platform}")
    df = load_data(args.platform)
    corpus = build_corpus(df)
    generate_wordcloud(corpus, args.platform, args.max_words)


if __name__ == "__main__":
    main()

