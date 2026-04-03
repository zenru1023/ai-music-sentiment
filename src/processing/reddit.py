"""
Reddit 댓글 전처리 스크립트
data/raw/reddit/ 의 CSV 파일들을 불러와 정제 후
data/processed/reddit_cleaned.csv 로 저장

처리 순서:
1. 전체 CSV 로드 및 병합
2. 무관한 댓글 제거 (포스트 제목 기반 키워드 필터 — 파일명에서 키워드 추출)
3. 중복 댓글 제거
4. 너무 짧은 댓글 제거 (3단어 미만 AND 15자 미만)
5. 이모지 제거
6. 소문자화
7. 스팸 필터링
8. 저장
"""

import os
import re
import glob
import pandas as pd

RAW_DIR = "data/raw/reddit"
OUTPUT_PATH = "data/processed/reddit_cleaned.csv"

# ── 1차 통과 조건: 파일명에서 추출한 키워드가 AI 음악 관련이어야 함 ──────────
# collect_reddit_json.py 의 KEYWORDS 목록과 대응
# 파일명 형식: rd_{subreddit}_{keyword_slug}_{timestamp}.csv
AI_MUSIC_KEYWORDS = {
    "ai_music", "ai_generated_music", "suno", "udio",
    "ai_music_debate", "ai_replacing_musicians", "ai_music_copyright",
    "suno_ai", "ai_music_generator", "ai_music_opinion",
    "ai_music_discussion", "ai_music_quality", "ai_music_creativity",
    "ai_music_future", "ai_music_industry", "ai_music_impact",
    "ai_music_innovation", "ai_song_generation", "ai_song_writing",
    # 수집 시 추가된 키워드
    "ai_music_ethics", "ai_music_feedback",
}

# ── 2차 제거 조건: 텍스트 내 명백히 무관한 맥락 ──────────────────────────────
# (코딩 AI, 금융 AI 등이 댓글에 혼입된 경우 대비)
TEXT_BLACKLIST_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(chatgpt|gpt-?4|claude|gemini|copilot)\b(?!.{0,50}(music|song|audio|sound|track|melody|beat))",
        # AI 코딩/텍스트 도구 언급이면서 음악 문맥이 없는 경우
    ]
]

# ── 삭제/제거된 댓글 패턴 ────────────────────────────────────────────────────
DELETED_PATTERN = re.compile(r"^\[(deleted|removed)\]$", re.IGNORECASE)

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)

SPAM_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"(.)\1{4,}",                        # 같은 문자 5회 이상 반복 (aaaa, !!!!)
        r"\b(subscribe|sub4sub|follow me|check out my|click here|link in bio)\b",
        r"\b(buy|cheap|discount|promo|sale|offer|deal|free)\b.{0,30}\b(now|today|here|click)\b",
        r"(first|1st)\s*[!😊🎉]*$",          # "first!" 류
        r"^\W+$",                             # 특수문자/이모지만 있는 댓글
    ]
]

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002300-\U000023FF"
    "\u2764\u2665\u2666\u2663\u2660"
    "]+",
    flags=re.UNICODE,
)


# ── 파일명에서 키워드 슬러그 추출 ─────────────────────────────────────────────

def extract_keyword_slug(filename: str) -> str:
    """
    파일명: rd_{subreddit}_{keyword_slug}_{timestamp}.csv
    예시:   rd_Music_ai_music_20240101_120000.csv → 'ai_music'

    타임스탬프는 숫자 8자리_숫자 6자리 형식이므로 역방향으로 제거.
    """
    name = os.path.basename(filename).replace(".csv", "")
    # rd_ 제거
    name = name[len("rd_"):]
    # 타임스탬프 제거 (_YYYYMMDD_HHMMSS)
    name = re.sub(r"_\d{8}_\d{6}$", "", name)
    # 서브레딧 제거 (첫 번째 _ 앞부분)
    parts = name.split("_", 1)
    return parts[1] if len(parts) > 1 else ""


def is_relevant_file(filename: str) -> bool:
    slug = extract_keyword_slug(filename).lower()
    return slug in AI_MUSIC_KEYWORDS


# ── 단계별 처리 함수 ─────────────────────────────────────────────────────────

def load_all_csvs(raw_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(raw_dir, "rd_*.csv"))
    if not files:
        raise FileNotFoundError(f"CSV 파일 없음: {raw_dir}")

    relevant_files = [f for f in files if is_relevant_file(f)]
    skipped = len(files) - len(relevant_files)

    print(f"로드 대상 — 전체 {len(files)}개 중 관련 {len(relevant_files)}개 ({skipped}개 스킵)")

    dfs = []
    for f in relevant_files:
        try:
            dfs.append(pd.read_csv(f, dtype=str))
        except Exception as e:
            print(f"  [!] 로드 실패: {f} — {e}")

    if not dfs:
        raise ValueError("로드 가능한 CSV 없음")

    df = pd.concat(dfs, ignore_index=True)
    print(f"로드 완료 — {len(relevant_files)}개 파일, {len(df):,}개 댓글")
    return df


def remove_deleted(df: pd.DataFrame) -> pd.DataFrame:
    """[deleted] / [removed] 댓글 제거 (수집 시 걸러지지만 혹시 남은 것 대비)"""
    before = len(df)
    mask = df["text"].fillna("").apply(lambda t: bool(DELETED_PATTERN.match(t.strip())))
    df = df[~mask].copy()
    removed = before - len(df)
    if removed:
        print(f"\n[단계 2] 삭제된 댓글 제거 — {before:,} → {len(df):,}개 ({removed:,}개 제거)")
    else:
        print(f"\n[단계 2] 삭제된 댓글 없음 — {len(df):,}개 유지")
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["comment_id"]).copy()
    print(f"\n[단계 3] 중복 제거 — {before:,} → {len(df):,}개 ({before - len(df):,}개 제거)")
    return df


def is_too_short(text: str) -> bool:
    """3단어 미만 AND 15자 미만이면 제거"""
    words = text.split()
    return len(words) < 3 and len(text) < 15


def filter_short(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    mask = df["text"].fillna("").apply(is_too_short)
    df = df[~mask].copy()
    print(f"\n[단계 4] 짧은 댓글 제거 — {before:,} → {len(df):,}개 ({before - len(df):,}개 제거)")
    return df


def remove_emoji(text: str) -> str:
    return EMOJI_PATTERN.sub("", text).strip()


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    df["text"] = (
        df["text"]
        .fillna("")
        .apply(remove_emoji)
        .str.lower()
        .str.strip()
    )
    print("\n[단계 5-6] 이모지 제거 + 소문자화 완료")
    return df


def is_spam(text: str) -> bool:
    if URL_PATTERN.search(text):
        return True
    return any(p.search(text) for p in SPAM_PATTERNS)


def filter_spam(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    mask = df["text"].apply(is_spam)
    df = df[~mask].copy()
    print(f"\n[단계 7] 스팸 필터링 — {before:,} → {len(df):,}개 ({before - len(df):,}개 제거)")
    return df


def drop_empty(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["text"].str.strip().str.len() > 0].copy()
    removed = before - len(df)
    if removed:
        print(f"\n[정리] 이모지 제거 후 빈 댓글 {removed:,}개 추가 제거")
    return df


def save(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료 → {path}")
    print(f"최종 댓글 수: {len(df):,}개")


def main():
    print("=" * 60)
    print("Reddit 댓글 전처리 시작")
    print("=" * 60)

    df = load_all_csvs(RAW_DIR)

    df = remove_deleted(df)
    df = drop_duplicates(df)
    df = filter_short(df)
    df = normalize_text(df)
    df = filter_spam(df)
    df = drop_empty(df)

    print("\n" + "=" * 60)
    save(df, OUTPUT_PATH)
    print("=" * 60)


if __name__ == "__main__":
    main()