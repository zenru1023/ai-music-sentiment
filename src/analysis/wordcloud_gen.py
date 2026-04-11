import argparse
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, STOPWORDS

# ── 1. process.py와 동일한 설정 적용 ──────────────────────────────────────────

# DTM 모드로 전처리된 데이터를 사용합니다.
PLATFORM_PATHS = {
    "youtube": "data/processed/dtm/youtube.csv",
    "reddit":  "data/processed/dtm/reddit.csv",
}
OUTPUT_DIR = "results/figures/wordcloud"

# process.py의 STOPWORDS 구성을 그대로 가져오거나 확장합니다.
# 워드클라우드는 시각적 방해 요소를 줄이기 위해 조금 더 엄격하게 필터링합니다.
CUSTOM_STOPWORDS = STOPWORDS | {
    # 구어체 및 필러 (process.py 참고)
    "lol", "lmao", "lmfao", "rofl", "omg", "wtf", "ngl", "tbh",
    "haha", "hehe", "yeah", "yep", "yup", "nah", "meh",
    "bro", "bruh", "dude", "guys", "man", "fam", "mate", "dawg",
    "just", "really", "actually", "literally", "basically", "honestly",
    "even", "still", "also", "well", "one", "thing", "something", 
    "way", "lot", "make", "made", "know", "think", "see", "going",
    
    # DTM 모드에서 제거하는 도메인 불용어 (ai, music 등)
    "ai", "music", "song", "track", "sound", "video", "youtube", "reddit"
}

def load_processed_data(platform: str):
    """
    process.py --mode dtm 과정을 거친 데이터를 로드합니다.
    이미 토큰화와 기본적인 불용어 제거가 완료된 상태입니다.
    """
    path = PLATFORM_PATHS[platform]
    if not os.path.exists(path):
        raise FileNotFoundError(f"전처리된 파일이 없습니다: {path}\n먼저 process.py를 실행하세요.")
    
    df = pd.read_csv(path, dtype=str)
    # NaN 제거 및 빈 문자열 필터링
    df = df[df["text"].notna() & (df["text"].str.strip() != "")].copy()
    
    # 이미 전처리가 되어있으므로 텍스트를 하나로 합치기만 하면 됩니다.
    full_text = " ".join(df["text"].tolist())
    print(f"[{platform.upper()}] 전처리 데이터 로드 완료: {len(df):,}개 댓글")
    return full_text

def generate_wordcloud(text: str, platform: str, max_words: int):
    """
    워드클라우드를 생성하고 저장합니다.
    """
    # process.py의 tokenize 규칙에 맞춰 정규식 적용 (알파벳+숫자만 추출)
    # 워드클라우드 라이브러리 내부 토큰화 대신, 우리가 정의한 텍스트를 그대로 사용하도록 설정
    wc = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        stopwords=CUSTOM_STOPWORDS,
        max_words=max_words,
        collocations=False,     # 이미 토큰화된 데이터이므로 단어 조합(Bigram)은 끕니다.
        colormap="viridis",     # 시각적 가독성
        prefer_horizontal=0.85,
        relative_scaling=0.5
    ).generate(text)

    # 캔버스 설정
    plt.figure(figsize=(16, 9))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud - {platform.capitalize()} (Mode: DTM)", fontsize=20, pad=20)

    # 결과 저장
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"wordcloud_{platform}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"워드클라우드 이미지 저장 완료 → {out_path}")

def main():
    parser = argparse.ArgumentParser(description="프로젝트 전처리 맞춤형 워드클라우드 생성")
    parser.add_argument("--platform", choices=["youtube", "reddit"], required=True)
    parser.add_argument("--max-words", type=int, default=150)
    args = parser.parse_args()

    print(f"[워드클라우드 생성] 플랫폼: {args.platform}")
    
    try:
        # 이미 process.py에서 모드별로 저장된 데이터를 불러오므로 별도 로직이 간소화됨
        corpus = load_processed_data(args.platform)
        
        if not corpus.strip():
            print("데이터가 비어 있어 워드클라우드를 생성할 수 없습니다.")
            return

        generate_wordcloud(corpus, args.platform, args.max_words)
        
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()