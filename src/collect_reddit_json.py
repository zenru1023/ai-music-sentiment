import os
import csv
import time
import random
import requests
from datetime import datetime, timezone

OUTPUT_DIR = "data/raw/reddit"

# 수집할 서브레딧 + 검색 키워드
SUBREDDITS = [
    "Music",
    "artificial",
    "WeAreTheMusicMakers",
    "MachineLearning",
    "Suno",
    "udiomusic",
]

KEYWORDS = [
    "AI music",
    "AI generated music",
    "Suno",
    "Udio",
    "AI music debate",
    "AI replacing musicians",
    "AI music copyright",
    "Suno AI",
    "AI music generator",
    "AI music opinion",
    "AI music discussion",
    "AI music quality",
    "AI music creativity",
    "AI music future",
    "AI music industry",
    "AI music impact",
    "AI music innovation",
    "AI song generation",
    "AI song writing"
]

MAX_POSTS =  100       # 키워드당 최대 포스트 수
MAX_COMMENTS = 1000    # 포스트당 최대 댓글 수

HEADERS = {"User-Agent": "ai_music_research/1.0"}


def already_collected(subreddit, keyword):
    if not os.path.exists(OUTPUT_DIR):
        return False
    slug = keyword.replace(" ", "_")
    return any(
        f.startswith(f"rd_{subreddit}_{slug}_")
        for f in os.listdir(OUTPUT_DIR)
    )


def get_collected_post_ids():
    """이미 수집된 post_id 전체 반환"""
    collected = set()
    if not os.path.exists(OUTPUT_DIR):
        return collected
    for fname in os.listdir(OUTPUT_DIR):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(OUTPUT_DIR, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "post_id" in row:
                        collected.add(row["post_id"])
        except Exception:
            continue
    return collected


def fetch_posts(subreddit, keyword, limit=100):
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    params = {
        "q": keyword,
        "restrict_sr": 1,
        "sort": "relevance",
        "limit": min(limit, 100),
        "t": "all",
    }
    try:
        res = requests.get(url, params=params, headers=HEADERS, timeout=10)
        res.raise_for_status()
        return res.json()["data"]["children"]
    except Exception as e:
        print(f"  [!] 포스트 수집 실패 ({subreddit} / {keyword}): {e}")
        return []


def fetch_comments(post_id, limit=200):
    url = f"https://www.reddit.com/comments/{post_id}.json"
    params = {"limit": limit, "depth": 1}
    try:
        res = requests.get(url, params=params, headers=HEADERS, timeout=10)
        res.raise_for_status()
        data = res.json()
        if len(data) < 2:
            return []
        comments = []
        for item in data[1]["data"]["children"]:
            if item["kind"] != "t1":
                continue
            d = item["data"]
            if not d.get("body") or d["body"] in ("[deleted]", "[removed]"):
                continue
            comments.append({
                "post_id":      post_id,
                "comment_id":   d.get("id", ""),
                "text":         d.get("body", ""),
                "author":       d.get("author", ""),
                "likes":        d.get("ups", 0),
                "published_at": datetime.fromtimestamp(
                                    d.get("created_utc", 0),
                                    tz=timezone.utc
                                ).isoformat(),
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "platform":     "reddit",
            })
        return comments
    except Exception as e:
        print(f"  [!] 댓글 수집 실패 (post: {post_id}): {e}")
        return []


def save_to_csv(rows, subreddit, keyword):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    slug = keyword.replace(" ", "_")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filepath = f"{OUTPUT_DIR}/rd_{subreddit}_{slug}_{timestamp}.csv"

    fieldnames = [
        "post_id", "comment_id", "text", "author",
        "likes", "published_at", "collected_at", "platform"
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return filepath


def main():
    total = 0
    collected_ids = get_collected_post_ids()  # 추가
    jobs = [(s, k) for s in SUBREDDITS for k in KEYWORDS]
    print(f"수집 시작 — {len(jobs)}개 조합")
    print(f"이미 수집된 포스트: {len(collected_ids)}개\n")

    for i, (subreddit, keyword) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] r/{subreddit} / '{keyword}'")

        if already_collected(subreddit, keyword):
            print(f"  → 이미 수집됨, 스킵\n")
            continue

        posts = fetch_posts(subreddit, keyword, MAX_POSTS)
        
        # 이미 수집된 포스트 필터링
        new_posts = [p for p in posts if p["data"]["id"] not in collected_ids]
        print(f"  → 포스트 {len(posts)}개 중 신규 {len(new_posts)}개")

        all_comments = []
        for post in new_posts:  # posts → new_posts
            post_id = post["data"]["id"]
            comments = fetch_comments(post_id, MAX_COMMENTS)
            all_comments.extend(comments)
            collected_ids.add(post_id)  # 실시간 업데이트
            time.sleep(random.uniform(3.0, 6.0))

        if all_comments:
            path = save_to_csv(all_comments, subreddit, keyword)
            total += len(all_comments)
            print(f"  → {len(all_comments)}개 댓글 저장 → {path}\n")
        else:
            print(f"  → 신규 포스트 없음\n")

        time.sleep(random.uniform(4.0, 7.0))

    print(f"완료 — 총 {total}개 댓글 수집")


if __name__ == "__main__":
    main()