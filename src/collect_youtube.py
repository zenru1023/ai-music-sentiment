import os
import time
import csv
from datetime import datetime, timezone
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

# 수집 동영상 ID
VIDEO_IDS = [
    "H3_E37znw18",
    "hMV3pH9rdHw",
    "ECLy6JnBdoY",
    "vA63-nDMYGg",
    "4He-MET8fik",
    "ZZ0BOEOtD2U",
    "Ey75Xw_ikqs",
    "jHrkQ928VNI",
    "2EKzWNcXrM0",
    "wBf6Yfahb2I",
    "a_qhYjVyawk",
    "QjHRZcFD6q4",
    "Z_2g0L3Hzgw",
    "1aZTpqKiH-M",
    "QtZDkgzjmQI",
    "z4wcpevDRYQ",
    "I_yLKjyl1N4",
    "rGremoYVMPc",
    "tJOUKpVPvsU",
    "CrcUJI197Vs",
    "U8dcFhF0Dlk",
    "1D4FAvqy8aQ",
    "FhhApqs2t-c",
    "3XVfWmpnc2A",
    "8yKrrPPEm10",
    "aQC0FI_asKY",
    "vY6FsswFIdY",
    "PaFZNuBXlEU",
    "eVlFcpX1VGA"
]

MAX_COMMENTS_PER_VIDEO = 5000
OUTPUT_DIR = "data/raw/youtube"


def get_youtube_client():
    return build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))


def already_collected(video_id):
    """이미 수집된 영상 ID인지 확인"""
    if not os.path.exists(OUTPUT_DIR):
        return False
    existing = os.listdir(OUTPUT_DIR)
    return any(f.startswith(f"yt_{video_id}_") for f in existing)


def fetch_comments(youtube, video_id, max_comments=500):
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        try:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                pageToken=next_page_token,
                textFormat="plainText",
                order="relevance"
            ).execute()

        except Exception as e:
            print(f"  [!] 에러 발생 (video: {video_id}): {e}")
            break

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "video_id":     video_id,
                "comment_id":   item["snippet"]["topLevelComment"]["id"],
                "author":       snippet.get("authorDisplayName", ""),
                "text":         snippet.get("textDisplay", ""),
                "likes":        snippet.get("likeCount", 0),
                "reply_count":  item["snippet"].get("totalReplyCount", 0),
                "published_at": snippet.get("publishedAt", ""),
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "platform":     "youtube",
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(0.5)

    return comments


def save_to_csv(comments, video_id):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{OUTPUT_DIR}/yt_{video_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"

    fieldnames = [
        "video_id", "comment_id", "author", "text",
        "likes", "reply_count", "published_at", "collected_at", "platform"
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comments)

    return filename


def main():
    youtube = get_youtube_client()
    total = 0

    print(f"수집 시작 — 영상 {len(VIDEO_IDS)}개\n")

    for i, video_id in enumerate(VIDEO_IDS, 1):
        print(f"[{i}/{len(VIDEO_IDS)}] video_id: {video_id}")

        # 중복 확인
        if already_collected(video_id):
            print(f"  → 이미 수집된 영상, 스킵\n")
            continue

        comments = fetch_comments(youtube, video_id, MAX_COMMENTS_PER_VIDEO)

        if comments:
            path = save_to_csv(comments, video_id)
            total += len(comments)
            print(f"  → {len(comments)}개 수집 완료 → {path}\n")
        else:
            print(f"  → 댓글 없음 또는 비활성화된 영상\n")

        time.sleep(1.0)

    print(f"완료 — 총 {total}개 댓글 수집")


if __name__ == "__main__":
    main()