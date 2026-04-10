import os
import time
import random
from datetime import datetime, timezone
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

VIDEO_IDS_FILE = "data/video_ids.txt"
MAX_RESULTS_PER_KEYWORD = 50  # search.list 1회당 최대 50개 (API 한도)

# 검색 키워드 목록
KEYWORDS = [
    "AI generated music",
    "AI music generation",
    "generative AI music",
    "suno ai music",
    "udio ai music",
    "AI music copyright",
    "AI music ethics",
    "human vs AI music",
    "Ai music controversy",
    "AI replacing musicians",

    "AI music is bad",
    "AI music is good",
    "AI music sucks",

    "AI made a song",
    "I tired AI music",
    "AI music reaction",
    "AI cover song"
]


def get_youtube_client():
    return build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))


def estimate_quota():
    """실행 전 예상 쿼터 소모량 출력"""
    units = len(KEYWORDS) * 100  # search.list = 100 units/call
    print("=" * 50)
    print("예상 쿼터 소모량")
    print(f"  키워드 수   : {len(KEYWORDS)}개")
    print(f"  search.list : {len(KEYWORDS)}회 × 100 units = {units} units")
    print(f"  일일 잔여   : 약 {10000 - units} units (댓글 수집용)")
    print("=" * 50)
    print()


def load_existing_ids():
    """video_ids.txt에서 기존 ID 목록 로드 (중복 방지용)"""
    if not os.path.exists(VIDEO_IDS_FILE):
        return set()
    with open(VIDEO_IDS_FILE, encoding="utf-8") as f:
        return {
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        }


def search_video_ids(youtube, keyword, max_results=50):
    """단일 키워드로 YouTube 검색 → video_id 목록 반환"""
    video_ids = []
    next_page_token = None
    collected = 0

    while collected < max_results:
        fetch = min(50, max_results - collected)
        try:
            response = youtube.search().list(
                part="snippet",
                q=keyword,
                type="video",
                relevanceLanguage="en",
                maxResults=fetch,
                pageToken=next_page_token,
            ).execute()
        except Exception as e:
            print(f"  [!] 에러 발생 (keyword='{keyword}'): {e}")
            break

        for item in response.get("items", []):
            video_id = item["id"].get("videoId")
            if video_id:
                video_ids.append(video_id)
                collected += 1

        next_page_token = response.get("nextPageToken")
        if not next_page_token or collected >= max_results:
            break

        time.sleep(random.uniform(0.5, 1.5))

    return video_ids


def append_ids_to_file(new_ids):
    """새 video_id를 video_ids.txt에 추가"""
    os.makedirs(os.path.dirname(VIDEO_IDS_FILE), exist_ok=True)
    with open(VIDEO_IDS_FILE, "a", encoding="utf-8") as f:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        f.write(f"\n# [자동 검색] {timestamp}\n")
        for video_id in new_ids:
            f.write(f"{video_id}\n")


def main():
    estimate_quota()

    youtube = get_youtube_client()
    existing_ids = load_existing_ids()
    print(f"기존 video_ids.txt 로드 — {len(existing_ids)}개\n")

    all_new_ids = []
    total_found = 0
    total_skipped = 0

    for i, keyword in enumerate(KEYWORDS, 1):
        print(f"[{i}/{len(KEYWORDS)}] 검색 중: '{keyword}'")
        found_ids = search_video_ids(youtube, keyword, MAX_RESULTS_PER_KEYWORD)
        total_found += len(found_ids)

        # 기존 ID 및 이번 수집 중 중복 제거
        new_ids = [
            vid for vid in found_ids
            if vid not in existing_ids and vid not in all_new_ids
        ]
        skipped = len(found_ids) - len(new_ids)
        total_skipped += skipped

        all_new_ids.extend(new_ids)
        existing_ids.update(new_ids)  # 이후 키워드에서도 중복 체크

        print(f"  → 발견 {len(found_ids)}개 / 신규 {len(new_ids)}개 / 중복 스킵 {skipped}개\n")

        if i < len(KEYWORDS):
            delay = random.uniform(2.0, 4.0)
            time.sleep(delay)

    # video_ids.txt에 추가
    if all_new_ids:
        append_ids_to_file(all_new_ids)
        print(f"완료 — 신규 {len(all_new_ids)}개 ID를 {VIDEO_IDS_FILE}에 추가")
        print(f"       (전체 발견 {total_found}개 중 중복 {total_skipped}개 제외)")
    else:
        print("신규 ID 없음 — video_ids.txt 변경 없음")

    print(f"\n다음 단계: python src/collect_youtube.py 실행")


if __name__ == "__main__":
    main()