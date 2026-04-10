import os
import json
import csv
from datetime import datetime, timezone

RAW_DIRS = {
    "youtube": "data/raw/youtube",
    "reddit":  "data/raw/reddit",
}
SUMMARY_PATH = "data/collection_summary.json"


def count_rows(filepath):
    try:
        with open(filepath, encoding="utf-8") as f:
            return sum(1 for _ in csv.reader(f)) - 1  # 헤더 제외
    except Exception:
        return 0


def summarize():
    summary = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "platforms": {},
        "total": 0,
    }

    for platform, directory in RAW_DIRS.items():
        if not os.path.exists(directory):
            summary["platforms"][platform] = {"files": 0, "comments": 0, "file_list": []}
            continue

        files = [f for f in os.listdir(directory) if f.endswith(".csv")]
        file_details = []
        platform_total = 0

        for fname in sorted(files):
            fpath = os.path.join(directory, fname)
            count = count_rows(fpath)
            platform_total += count
            file_details.append({"file": fname, "comments": count})

        summary["platforms"][platform] = {
            "files":     len(files),
            "comments":  platform_total,
            "file_list": file_details,
        }
        summary["total"] += platform_total

    # 저장
    os.makedirs("data", exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 출력
    print(f"=== 데이터 수집 현황 ({summary['updated_at']}) ===\n")
    for platform, info in summary["platforms"].items():
        print(f"[{platform}]")
        print(f"  파일 수  : {info['files']}개")
        print(f"  댓글 수  : {info['comments']:,}개")
        for f in info["file_list"]:
            print(f"    - {f['file']} ({f['comments']:,}개)")
        print()
    print(f"총 수집량 : {summary['total']:,}개")
    print(f"저장 위치 : {SUMMARY_PATH}")


if __name__ == "__main__":
    summarize()