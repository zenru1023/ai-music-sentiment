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
8. 불용어 제거 (구어체 호칭 / 필러 / 인터넷 슬랭)
9. 불용어 제거 후 짧아진 댓글 재필터링
10. 비영어 토큰 제거 (oov 필터 — typo / 비영어 알파벳 토큰)
11. 비영어 토큰 제거 후 짧아진 댓글 재필터링
12. 저장
"""

import os
import re
import glob
import pandas as pd
import nltk
from nltk.corpus import words as nltk_words_corpus

# nltk words 코퍼스 자동 다운로드 (최초 1회)
try:
    nltk_words_corpus.words()
except LookupError:
    nltk.download("words", quiet=True)

RAW_DIR = "data/raw/reddit"
OUTPUT_PATH = "data/processed/reddit_cleaned.csv"

# ── 1차 통과 조건: 파일명에서 추출한 키워드가 AI 음악 관련이어야 함 ──────────
AI_MUSIC_KEYWORDS = {
    "ai_music", "ai_generated_music", "suno", "udio",
    "ai_music_debate", "ai_replacing_musicians", "ai_music_copyright",
    "suno_ai", "ai_music_generator", "ai_music_opinion",
    "ai_music_discussion", "ai_music_quality", "ai_music_creativity",
    "ai_music_future", "ai_music_industry", "ai_music_impact",
    "ai_music_innovation", "ai_song_generation", "ai_song_writing",
    "ai_music_ethics", "ai_music_feedback",
}

# ── 2차 제거 조건: 텍스트 내 명백히 무관한 맥락 ──────────────────────────────
TEXT_BLACKLIST_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(chatgpt|gpt-?4|claude|gemini|copilot)\b(?!.{0,50}(music|song|audio|sound|track|melody|beat))",
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
# - AI 음악 관련 고유명사 / 브랜드명
# - 음악 장르·용어
# - 숫자 포함 토큰은 코드에서 별도 처리 (정규식)
WHITELIST: set[str] = {
    # AI 음악 도구 고유명사
    "suno", "udio", "musiclm", "mubert", "aiva", "boomy", "soundraw",
    "udiomusic", "sunoai",
    # 음악 장르 / 용어 (사전에 없을 수 있음)
    "lo-fi", "lofi", "hiphop", "edm", "dnb", "dubstep", "synthwave",
    "hyperpop", "glitchcore", "vaporwave", "chiptune", "phonk",
    "808s", "808", "bpm", "daw", "vst", "midi", "wav", "mp3", "flac",
    # 인터넷 구어체 중 의미 있는 것 (불용어 목록과 중복 없이)
    "gonna", "wanna", "kinda", "sorta", "gotta", "dunno",
    "yeah", "yea", "nah",
    # 기타 자주 쓰이는 약어
    "ai", "ml", "llm", "tts", "api", "ui", "ux",
    "vs", "etc", "aka",
}

# 숫자가 포함된 토큰은 화이트리스트 여부와 무관하게 유지 (4k, 808s, mp3 등)
_HAS_DIGIT = re.compile(r"\d")


# ── 파일명에서 키워드 슬러그 추출 ─────────────────────────────────────────────

def extract_keyword_slug(filename: str) -> str:
    name = os.path.basename(filename).replace(".csv", "")
    name = name[len("rd_"):]
    name = re.sub(r"_\d{8}_\d{6}$", "", name)
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

    # 불용어 제거 후 짧아진 댓글 재필터링
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
      4. 순수 숫자 (연도, 수치 등)
      5. 구두점만으로 구성된 토큰 (문장 부호 — 별도 처리하지 않음)
    """
    t = token.lower()
    if t in _ENGLISH_WORDS:
        return True
    if t in WHITELIST:
        return True
    if _HAS_DIGIT.search(t):
        return True
    if not t.isalpha():          # 구두점 포함 토큰 (예: 's, n't) 유지
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
    print("Reddit 댓글 전처리 시작")
    print("=" * 60)

    df = load_all_csvs(RAW_DIR)

    df = remove_deleted(df)
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