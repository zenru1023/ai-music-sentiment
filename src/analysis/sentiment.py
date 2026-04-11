import argparse
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── 경로 설정 ─────────────────────────────────────────

PLATFORM_PATHS = {
    "youtube": "data/processed/sentiment/youtube.csv",
    "reddit":  "data/processed/sentiment/reddit.csv",
}
OUTPUT_TABLES = "results/tables"
OUTPUT_FIGURES = "results/figures/sentiment"

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


# ── 데이터 로드 ───────────────────────────────────────

def load_data(platform: str) -> pd.DataFrame:
    path = PLATFORM_PATHS[platform]
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일 없음: {path}")
    df = pd.read_csv(path, dtype=str)
    df = df[df["text"].notna() & (df["text"].str.strip() != "")].copy()
    df["text"] = df["text"].str.strip()
    print(f"로드 완료 — {len(df):,}개 댓글")
    return df


# ── RoBERTa ──────────────────────────────────────────

def run_transformers(df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    print(f"\n[RoBERTa] 모델 로드 중 — {MODEL_NAME}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  디바이스: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    texts = df["text"].tolist()
    all_labels, all_scores = [], []

    print(f"  댓글 {len(texts):,}개 처리 중 (배치 크기: {batch_size})...")

    for i in tqdm(range(0, len(texts), batch_size), desc="RoBERTa Inference"):
        batch = texts[i:i + batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).cpu()

        for prob in probs:
            idx = int(prob.argmax().item())
            all_labels.append(LABEL_MAP[idx])

            all_scores.append({
                "roberta_neg": round(float(prob[0].item()), 4),
                "roberta_neu": round(float(prob[1].item()), 4),
                "roberta_pos": round(float(prob[2].item()), 4),
            })

    df["roberta_label"] = all_labels
    df["roberta_neg"]   = [s["roberta_neg"] for s in all_scores]
    df["roberta_neu"]   = [s["roberta_neu"] for s in all_scores]
    df["roberta_pos"]   = [s["roberta_pos"] for s in all_scores]

    dist = pd.Series(all_labels).value_counts()
    print(f"  positive: {dist.get('positive', 0):,}  "
          f"neutral: {dist.get('neutral', 0):,}  "
          f"negative: {dist.get('negative', 0):,}")

    return df


# ── 시각화 (RoBERTa만) ───────────────────────────────

def plot_sentiment(df: pd.DataFrame, platform: str):
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = {"positive": "#4CAF50", "neutral": "#9E9E9E", "negative": "#F44336"}
    order = ["positive", "neutral", "negative"]

    counts = df["roberta_label"].value_counts().reindex(order, fill_value=0)

    ax.bar(order, counts.values,
           color=[colors[l] for l in order], edgecolor="white")

    ax.set_title(f"RoBERTa — {platform.capitalize()}", fontsize=13)
    ax.set_ylabel("댓글 수")

    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01,
                f"{v:,}\n({v/len(df)*100:.1f}%)",
                ha="center", fontsize=9)

    plt.tight_layout()
    os.makedirs(OUTPUT_FIGURES, exist_ok=True)

    out_path = os.path.join(OUTPUT_FIGURES, f"sentiment_{platform}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n그래프 저장 → {out_path}")


# ── 저장 ─────────────────────────────────────────────

def save_results(df: pd.DataFrame, platform: str):
    os.makedirs(OUTPUT_TABLES, exist_ok=True)

    detail_path = os.path.join(OUTPUT_TABLES, f"sentiment_{platform}.csv")
    df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"상세 결과 저장 → {detail_path}")

    summary = {}

    for label in ["positive", "neutral", "negative"]:
        n = (df["roberta_label"] == label).sum()
        summary[f"roberta_{label}"] = n
        summary[f"roberta_{label}_pct"] = round(n / len(df) * 100, 2)

    summary_df = pd.DataFrame([summary])

    summary_path = os.path.join(
        OUTPUT_TABLES, f"sentiment_{platform}_summary.csv"
    )
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"요약 결과 저장 → {summary_path}")


# ── 메인 ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="감정 분석 — RoBERTa")
    parser.add_argument("--platform", choices=["youtube", "reddit"], required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    print(f"[감정 분석] 플랫폼: {args.platform}")

    df = load_data(args.platform)
    df = run_transformers(df, args.batch_size)

    plot_sentiment(df, args.platform)
    save_results(df, args.platform)

    print("\n완료.")


if __name__ == "__main__":
    main()