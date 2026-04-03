"""
YouTube 댓글 전처리 스크립트
data/raw/youtube/ 의 CSV 파일들을 불러와 정제 후
data/processed/youtube_cleaned.csv 로 저장

처리 순서:
1. 전체 CSV 로드 및 병합
2. 무관한 video_id 제거 (YouTube API title — AI 포함 + blacklist 제외)
3. 중복 댓글 제거
4. 너무 짧은 댓글 제거 (3단어 미만 AND 15자 미만)
5. 이모지 제거
6. 소문자화
7. 스팸 필터링
8. 저장
"""

import os
import re
import time
import glob
import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = "data/raw/youtube"
OUTPUT_PATH = "data/processed/youtube_cleaned.csv"

# ── 1차 통과 조건: 제목에 "ai" 또는 AI 음악 관련 고유명사 포함 ──────────────
# 대소문자 무관하게 매칭 (title.lower() 기준)
TITLE_AI_KEYWORDS = [
    " ai ", " ai,", " ai.", " ai!", " ai?", " ai:", " ai-",  # 단어 경계 처리
    "ai-", "(ai)", "[ai]",
    "a.i.", "artificial intelligence",
    "suno", "udio", "musiclm", "mubert", "aiva", "boomy", "soundraw",
    "voice clone", "voice cloning", "deepfake",
]

# ── 2차 제거 조건: 제목에 아래 키워드가 있으면 무관 영상으로 판단 ────────────
# (AI가 포함돼도 음악과 무관한 경우)
TITLE_BLACKLIST = [
    "cursor", "composer",          # 코딩 IDE
    "trading", "invest", "stock",  # 금융
    "coding", "code ", "developer", "programmer", "software", "github",
    "full movie", "film compilation",
    "baby", "cute baby",
    "essay on", "essay about",     # 영어 에세이 교육
    "asmr",
    "게임", "게임",                 # 한국어 게임 콘텐츠
]

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


def load_all_csvs(raw_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(raw_dir, "yt_*.csv"))
    if not files:
        raise FileNotFoundError(f"CSV 파일 없음: {raw_dir}")

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, dtype=str))
        except Exception as e:
            print(f"  [!] 로드 실패: {f} — {e}")

    df = pd.concat(dfs, ignore_index=True)
    print(f"로드 완료 — {len(files)}개 파일, {len(df):,}개 댓글")
    return df


def fetch_video_titles(video_ids: list[str]) -> dict[str, str]:
    """YouTube API videos.list로 제목 일괄 조회 (50개씩 배치)"""
    youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))
    titles = {}

    batches = [video_ids[i:i+50] for i in range(0, len(video_ids), 50)]
    for batch in batches:
        try:
            resp = youtube.videos().list(
                part="snippet",
                id=",".join(batch),
            ).execute()
            for item in resp.get("items", []):
                vid = item["id"]
                title = item["snippet"].get("title", "")
                titles[vid] = title
        except Exception as e:
            print(f"  [!] API 에러: {e}")
        time.sleep(0.3)

    return titles


def is_relevant(title: str) -> bool:
    """
    관련 영상 판단 기준:
    - 제목에 "ai" 또는 AI 관련 고유명사가 포함되어야 함 (1차 통과)
    - blacklist 키워드가 있으면 제거 (2차 필터)
    """
    t = title.lower()

    # 1차: AI 키워드 포함 여부
    has_ai = any(kw in t for kw in TITLE_AI_KEYWORDS)
    # 제목이 "ai"로 시작하거나 끝나는 경우도 처리
    if not has_ai:
        has_ai = t.startswith("ai ") or t.endswith(" ai")
    if not has_ai:
        return False

    # 2차: blacklist 키워드가 있으면 제거
    is_blacklisted = any(kw in t for kw in TITLE_BLACKLIST)
    return not is_blacklisted


def filter_by_title(df: pd.DataFrame) -> pd.DataFrame:
    video_ids = df["video_id"].unique().tolist()
    print(f"\n[단계 2] 무관한 영상 제거 — {len(video_ids)}개 video_id 조회 중...")

    titles = fetch_video_titles(video_ids)

    # API에서 가져오지 못한 ID는 일단 유지 (삭제된 영상 등)
    relevant_ids = {
        vid for vid, title in titles.items() if is_relevant(title)
    }
    missing_ids = set(video_ids) - set(titles.keys())
    relevant_ids |= missing_ids  # 조회 실패한 건 보수적으로 유지

    removed_ids = set(video_ids) - relevant_ids
    if removed_ids:
        print(f"  제거된 video_id ({len(removed_ids)}개):")
        for vid in sorted(removed_ids):
            print(f"    - {vid}: {titles.get(vid, '(제목 없음)')}")

    before = len(df)
    df = df[df["video_id"].isin(relevant_ids)].copy()
    print(f"  {before:,} → {len(df):,}개 ({before - len(df):,}개 제거)")
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
    print("YouTube 댓글 전처리 시작")
    print("=" * 60)

    df = load_all_csvs(RAW_DIR)

    df = filter_by_title(df)
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