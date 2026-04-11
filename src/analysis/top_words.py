import argparse
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# ── 1. process.py의 경로 및 설정 준수 ────────────────────────────────────────

PLATFORM_PATHS = {
    "youtube": "data/processed/dtm/youtube.csv",
    "reddit":  "data/processed/dtm/reddit.csv",
}
OUTPUT_DIR = "results/figures/top_words"
RESULTS_DIR = "results/tables/top_words" # 통일성을 위해 폴더 구조 세분화

# process.py의 STOPWORDS 구성을 반영
# 이미 dtm 모드에서 처리되었으나, 혹시 모를 누락이나 추가 필터링을 위해 정의
STOPWORDS = {
    # 일반적인 의미 없는 단어들 (필요 시 추가)
    "get", "got", "make", "made", "know", "think", "see", "going", "would", "could"
}

def load_data(platform: str) -> pd.DataFrame:
    path = PLATFORM_PATHS[platform]
    if not os.path.exists(path):
        raise FileNotFoundError(f"전처리된 파일 없음: {path}\n먼저 'process.py --mode dtm'을 실행하세요.")
    
    df = pd.read_csv(path, dtype=str)
    # NaN 및 빈 문자열 제거
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]
    print(f"[{platform.upper()}] 로드 완료 — {len(df):,}개 댓글")
    return df

def get_tokens(df: pd.DataFrame) -> list[str]:
    """
    process.py에서 이미 토큰화되어 공백으로 합쳐진 'text' 컬럼을 
    다시 분리하여 리스트로 만듭니다.
    """
    all_tokens = []
    for text in df["text"]:
        # 이미 process.py에서 \b[a-zA-Z0-9]+\b 규칙으로 정제된 상태임
        words = str(text).split()
        all_tokens.extend(w for w in words if w not in STOPWORDS)
    return all_tokens

def plot_top_words(counter: Counter, top_n: int, platform: str):
    # 상위 N개 추출
    top_data = counter.most_common(top_n)
    if not top_data:
        print("시각화할 데이터가 없습니다.")
        return
        
    words, counts = zip(*top_data)

    # 시각화 (깔끔한 수평 바 차트)
    fig, ax = plt.subplots(figsize=(12, 8))
    # 상위 단어가 위로 오도록 리스트 반전
    y_pos = range(len(words))
    bars = ax.barh(y_pos[::-1], counts, color="skyblue", edgecolor="navy", alpha=0.7)

    # 바 끝에 숫자 표시
    for i, count in enumerate(counts):
        ax.text(
            count + (max(counts) * 0.01), 
            len(words) - 1 - i, 
            f"{count:,}", 
            va="center", fontsize=10, fontweight='bold'
        )

    ax.set_yticks(y_pos[::-1])
    ax.set_yticklabels(words, fontsize=11)
    ax.set_xlabel("Frequency (Count)", fontsize=12, labelpad=10)
    ax.set_title(f"Top {top_n} Keywords - {platform.capitalize()} (Mode: DTM)", fontsize=15, pad=20)
    
    # 천 단위 콤마 포맷
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"top_words_{platform}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"그래프 저장 완료 → {out_path}")

def save_results(counter: Counter, top_n: int, platform: str):
    # CSV 저장
    rows = [{"rank": i+1, "word": w, "count": c}
            for i, (w, c) in enumerate(counter.most_common(top_n))]
    df_out = pd.DataFrame(rows)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"top_words_{platform}.csv")
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV 결과 저장 완료 → {csv_path}")

    # 터미널 출력
    print(f"\n[Top {top_n} 분석 결과]")
    print("-" * 40)
    for _, row in df_out.head(10).iterrows(): # 상위 10개만 살짝 보여줌
        print(f" {int(row['rank']):>2}. {row['word']:<15} : {int(row['count']):>8,}")

def main():
    parser = argparse.ArgumentParser(description="프로젝트 맞춤형 상위 단어 분석")
    parser.add_argument("--platform", choices=["youtube", "reddit"], required=True)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    # 1. 데이터 로드 (dtm 모드 파일)
    df = load_data(args.platform)
    
    # 2. 토큰 집계
    tokens = get_tokens(df)
    print(f"총 분석 토큰 수: {len(tokens):,}개")
    
    counter = Counter(tokens)
    
    # 3. 시각화 및 저장
    plot_top_words(counter, args.top_n, args.platform)
    save_results(counter, args.top_n, args.platform)

if __name__ == "__main__":
    main()