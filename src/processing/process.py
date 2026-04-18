"""
uv run src/processing/process.py --platform youtube or reddit --mode sentiment or topic or dtm
"""

import os
import re
import glob
import argparse
import pandas as pd
import nltk

from itertools import combinations
from collections import Counter
from nltk.corpus import stopwords

# =========================
# NLTK 다운로드
# =========================
try:
    stopwords.words("english")
except:
    nltk.download("stopwords")

# =========================
# 경로 설정
# =========================

RAW_DIR = {
    "reddit": "data/raw/reddit",
    "youtube": "data/raw/youtube"
}

BASE_OUTPUT_DIR = "data/processed"


# =========================
# STOPWORDS
# =========================

BASE_STOPWORDS = set(stopwords.words("english"))

COMMON_STOPWORDS = {
    # 인터넷 슬랭
    "lol", "lmao", "lmfao", "rofl", "omg", "wtf", "ngl", "tbh",
    "imo", "imho", "smh", "idk", "btw", "rn", "irl",

    # 감탄/웃음
    "haha", "hehe", "yeah", "yep", "yup", "nah", "meh",

    # 구어체 호칭
    "bro", "bruh", "dude", "guys", "man", "fam", "mate", "dawg",

    # 필러 부사 (like는 제외 권장)
    "just", "really", "actually", "literally",
    "basically", "honestly", "seriously",
    "even", "still", "also", "well",

    # 의미 약한 일반어
    "one", "thing", "something", "anything",
    "way", "lot", "kind", "sort",
}

DOMAIN_STOPWORDS_TOPIC = {
    "people","thing","something","anything",
    "someone","everyone","stuff"
}

DOMAIN_STOPWORDS_DTM = {
    "ai","music","song","track","sound"
}

# =========================
# SPAM 필터
# =========================

SPAM_PATTERNS = [
    re.compile(r"(.)\1{4,}"),
    re.compile(r"\b(subscribe|follow me|click here|check out)\b", re.IGNORECASE),
    re.compile(r"\b(buy|cheap|discount|promo)\b.*\b(now|click)\b", re.IGNORECASE),
]

def is_spam(text):
    if re.search(r"http", text):
        return True
    for p in SPAM_PATTERNS:
        if p.search(text):
            return True
    if len(text) < 10:
        return True
    return False

# =========================
# 유틸
# =========================

def normalize_text(text):
    text = str(text)

    # emoji 제거
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # 소문자
    text = text.lower()

    # 반복 문자 축소 (loooool → lool)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    return text.strip()

def tokenize(text):
    return re.findall(r"\b[a-zA-Z0-9]+\b", text)

def remove_stopwords(tokens, stopword_set):
    return [t for t in tokens if t not in stopword_set]

def is_short(tokens):
    return len(tokens) < 4

# =========================
# 데이터 로드
# =========================

def load_reddit():
    files = glob.glob(os.path.join(RAW_DIR["reddit"], "rd_*.csv"))

    if not files:
        raise FileNotFoundError("Reddit CSV 파일 없음")

    print(f"로드 파일 수: {len(files)}")

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, dtype=str))
        except Exception as e:
            print(f"[!] 로드 실패: {f} — {e}")

    df = pd.concat(dfs, ignore_index=True)

    print(f"총 댓글 수: {len(df):,}")
    return df

def load_youtube():
    files = glob.glob(os.path.join(RAW_DIR["youtube"], "yt_*.csv"))
    dfs = [pd.read_csv(f, dtype=str) for f in files]
    return pd.concat(dfs, ignore_index=True)

# =========================
# 전처리
# =========================

def preprocess(df, mode):
    print(f"초기 데이터: {len(df):,}")

    # 공통 전처리
    df["text"] = df["text"].fillna("").apply(normalize_text)

    # 스팸 제거
    df = df[~df["text"].apply(is_spam)]

    # 중복 제거 (comment_id 기준)
    if "comment_id" in df.columns:
        df = df.drop_duplicates(subset=["comment_id"])

    # =========================
    # MODE 분기
    # =========================

    # -------------------------
    # 1. SENTIMENT
    # -------------------------
    if mode == "sentiment":
        df = df[df["text"].str.len() > 0]

    # -------------------------
    # 2. TOPIC (LDA)
    # -------------------------
    elif mode == "topic":
        stopword_set = BASE_STOPWORDS | COMMON_STOPWORDS | DOMAIN_STOPWORDS_TOPIC

        df["tokens"] = df["text"].apply(tokenize)
        df["tokens"] = df["tokens"].apply(lambda x: remove_stopwords(x, stopword_set))
        df = df[~df["tokens"].apply(is_short)]

        df["text"] = df["tokens"].apply(lambda x: " ".join(x))

        # topic에서는 text 중복 제거 가능
        df = df.drop_duplicates(subset=["text"])

    # -------------------------
    # 3. DTM / WordCloud
    # -------------------------
    elif mode == "dtm":
        stopword_set = BASE_STOPWORDS | COMMON_STOPWORDS | DOMAIN_STOPWORDS_TOPIC | DOMAIN_STOPWORDS_DTM

        df["tokens"] = df["text"].apply(tokenize)
        df["tokens"] = df["tokens"].apply(lambda x: remove_stopwords(x, stopword_set))
        df = df[~df["tokens"].apply(is_short)]

        df["text"] = df["tokens"].apply(lambda x: " ".join(x))

        df = df.drop_duplicates(subset=["text"])

    # =========================

    print(f"최종 데이터: {len(df):,}")
    return df

# =========================
# 실행
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True, choices=["reddit", "youtube"])
    parser.add_argument("--mode", required=True, choices=["sentiment", "topic", "dtm"])

    args = parser.parse_args()

    # 데이터 로드
    if args.platform == "reddit":
        df = load_reddit()
    else:
        df = load_youtube()

    # 전처리
    df = preprocess(df, args.mode)

    # 저장
    out = f"{BASE_OUTPUT_DIR}/{args.mode}/{args.platform}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"저장 완료 → {out}")

# =========================

if __name__ == "__main__":
    main()