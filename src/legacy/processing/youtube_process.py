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
8. 불용어 제거 (구어체 호칭 / 필러 / 인터넷 슬랭)
9. 불용어 제거 후 짧아진 댓글 재필터링
10. 비영어 토큰 제거 (oov 필터 — typo / 비영어 알파벳 토큰)
11. 비영어 토큰 제거 후 짧아진 댓글 재필터링
12. 저장
"""

import os
import re
import time
import glob
import pandas as pd
import nltk
from nltk.corpus import words as nltk_words_corpus
from googleapiclient.discovery import build
from dotenv import load_dotenv

# nltk words 코퍼스 자동 다운로드 (최초 1회)
try:
    nltk_words_corpus.words()
except LookupError:
    nltk.download("words", quiet=True)

load_dotenv()

RAW_DIR = "data/raw/youtube"
OUTPUT_PATH = "data/processed/youtube_cleaned.csv"

# ── 1차 통과 조건: 제목에 "ai" 또는 AI 음악 관련 고유명사 포함 ──────────────
TITLE_AI_KEYWORDS = [
    " ai ", " ai,", " ai.", " ai!", " ai?", " ai:", " ai-",
    "ai-", "(ai)", "[ai]",
    "a.i.", "artificial intelligence",
    "suno", "udio", "musiclm", "mubert", "aiva", "boomy", "soundraw",
    "voice clone", "voice cloning", "deepfake",
]

# ── 2차 제거 조건: 제목에 아래 키워드가 있으면 무관 영상으로 판단 ────────────
TITLE_BLACKLIST = [
    "cursor", "composer",
    "trading", "invest", "stock",
    "coding", "code ", "developer", "programmer", "software", "github",
    "full movie", "film compilation",
    "baby", "cute baby",
    "essay on", "essay about",
    "asmr",
    "게임", "게임",
]

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)

SPAM_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"(.)\1{4,}",
        r"\b(subscribe|sub4sub|follow me|check out my|click here|link in bio)\b",
        r"\b(buy|cheap|discount|promo|sale|offer|deal|free)\b.{0,30}\b(now|today|here|click)\b",
        r"(first|1st)\s*[!😊🎉]*$",
        r"^\W+$",
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

# ── 불용어 목록 ───────────────────────────────────────────────────────────────
# normalize_text(소문자화) 이후에 적용되므로 모두 소문자로 정의
STOPWORDS = {
    # 구어체 호칭
    "bro", "bruh", "brah", "sis", "dude", "man", "dog", "dawg",
    "homie", "mate", "fam", "guys", "buddy", "pal",
    # 감탄사 / 필러
    "lol", "lmao", "lmfao", "rofl", "omg", "omfg", "wtf", "wth",
    "ngl", "tbh", "imo", "imho", "smh", "oof", "welp", "yikes",
    "fr", "fwiw", "iirc", "afaik", "istg", "ikr", "idk",
    # 인터넷 슬랭 / 리액션
    "lmk", "btw", "rn", "irl", "imo", "aka",
    "haha", "hahaha", "hehe", "lmaooo", "looool",
    "yep", "yup", "nope", "nah", "meh",
}

# 단어 경계 기반 불용어 제거 패턴 (소문자화 이후 적용)
_STOPWORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in STOPWORDS) + r")\b",
    re.IGNORECASE,
)

# ── OOV(Out-of-Vocabulary) 필터 ───────────────────────────────────────────────
# nltk words 코퍼스 기반 영어 사전 (소문자 집합)
_ENGLISH_WORDS: set[str] = set(w.lower() for w in nltk_words_corpus.words())

# 사전에 없어도 유효한 토큰으로 인정하는 화이트리스트
WHITELIST: set[str] = {
    # AI 음악 도구 고유명사
    "suno", "udio", "musiclm", "mubert", "aiva", "boomy", "soundraw",
    "udiomusic", "sunoai",
    # 음악 장르 / 용어
    "lo-fi", "lofi", "hiphop", "edm", "dnb", "dubstep", "synthwave",
    "hyperpop", "glitchcore", "vaporwave", "chiptune", "phonk",
    "808s", "808", "bpm", "daw", "vst", "midi", "wav", "mp3", "flac",
    # 인터넷 구어체 중 의미 있는 것
    "gonna", "wanna", "kinda", "sorta", "gotta", "dunno",
    "yeah", "yea", "nah",
    # 기타 자주 쓰이는 약어
    "ai", "ml", "llm", "tts", "api", "ui", "ux",
    "vs", "etc", "aka",
}

# 숫자가 포함된 토큰은 화이트리스트 여부와 무관하게 유지 (4k, 808s, mp3 등)
_HAS_DIGIT = re.compile(r"\d")


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
    t = title.lower()
    has_ai = any(kw in t for kw in TITLE_AI_KEYWORDS)
    if not has_ai:
        has_ai = t.startswith("ai ") or t.endswith(" ai")
    if not has_ai:
        return False
    is_blacklisted = any(kw in t for kw in TITLE_BLACKLIST)
    return not is_blacklisted


def filter_by_title(df: pd.DataFrame) -> pd.DataFrame:
    video_ids = df["video_id"].unique().tolist()
    print(f"\n[단계 2] 무관한 영상 제거 — {len(video_ids)}개 video_id 조회 중...")

    titles = fetch_video_titles(video_ids)

    relevant_ids = {
        vid for vid, title in titles.items() if is_relevant(title)
    }
    missing_ids = set(video_ids) - set(titles.keys())
    relevant_ids |= missing_ids

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


def remove_stopwords(text: str) -> str:
    """불용어를 제거하고 다중 공백을 정리한다."""
    cleaned = _STOPWORD_PATTERN.sub("", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def filter_stopwords(df: pd.DataFrame) -> pd.DataFrame:
    """
    [단계 8] 불용어 토큰 제거
    [단계 9] 제거 후 너무 짧아진 댓글 재필터링 (is_too_short 기준 동일)
    """
    before = len(df)
    df["text"] = df["text"].apply(remove_stopwords)
    print(f"\n[단계 8] 불용어 제거 완료 ({len(STOPWORDS)}개 토큰 대상)")

    mask = df["text"].fillna("").apply(is_too_short)
    df = df[~mask].copy()
    removed = before - len(df)
    print(f"[단계 9] 불용어 제거 후 재필터링 — {before:,} → {len(df):,}개 ({removed:,}개 제거)")
    return df


def is_valid_token(token: str) -> bool:
    """
    토큰이 유효한 영어 단어인지 판별.
    유효 조건 (하나라도 해당하면 유지):
      1. nltk 영어 사전에 있음
      2. 화이트리스트에 있음
      3. 숫자를 포함함 (4k, 808s, mp3 등)
      4. 구두점 포함 토큰 (예: 's, n't) — 별도 처리하지 않음
    """
    t = token.lower()
    if t in _ENGLISH_WORDS:
        return True
    if t in WHITELIST:
        return True
    if _HAS_DIGIT.search(t):
        return True
    if not t.isalpha():
        return True
    return False


def remove_oov_tokens(text: str) -> str:
    """비영어 / oov 토큰을 제거하고 다중 공백을 정리한다."""
    tokens = text.split()
    kept = [tok for tok in tokens if is_valid_token(tok)]
    return re.sub(r"\s{2,}", " ", " ".join(kept)).strip()


def filter_oov(df: pd.DataFrame) -> pd.DataFrame:
    """
    [단계 10] 비영어 토큰 제거 (typo / 비영어 알파벳 토큰)
    [단계 11] 제거 후 너무 짧아진 댓글 재필터링
    """
    before = len(df)
    df["text"] = df["text"].apply(remove_oov_tokens)
    print(f"\n[단계 10] 비영어 토큰 제거 완료 (사전: nltk words {len(_ENGLISH_WORDS):,}개 + 화이트리스트 {len(WHITELIST)}개)")

    mask = df["text"].fillna("").apply(is_too_short)
    df = df[~mask].copy()
    removed = before - len(df)
    print(f"[단계 11] 비영어 토큰 제거 후 재필터링 — {before:,} → {len(df):,}개 ({removed:,}개 제거)")
    return df


def drop_empty(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["text"].str.strip().str.len() > 0].copy()
    removed = before - len(df)
    if removed:
        print(f"\n[정리] 빈 댓글 {removed:,}개 추가 제거")
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
    df = filter_stopwords(df)
    df = filter_oov(df)
    df = drop_empty(df)

    print("\n" + "=" * 60)
    save(df, OUTPUT_PATH)
    print("=" * 60)


if __name__ == "__main__":
    main()