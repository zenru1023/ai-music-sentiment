"""
AI 음악 감성 분석 시각화 스크립트

실행:
    uv run python src/visualize.py --platform all
    uv run python src/visualize.py --platform youtube
    uv run python src/visualize.py --platform reddit
"""

import argparse
import os
import warnings

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.models import LdaMulticore, Word2Vec
from gensim import corpora
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR    = os.path.join(BASE_DIR, "results", "models")
TABLES_DIR    = os.path.join(BASE_DIR, "results", "tables")
FIGURES_DIR   = os.path.join(BASE_DIR, "results", "figures")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(FIGURES_DIR, exist_ok=True)

# ── 공통 색상 ──────────────────────────────────────────────────────────────────
SENTIMENT_COLORS = {
    "positive": "#1D9E75",
    "neutral":  "#888780",
    "negative": "#D85A30",
}
PLATFORM_COLORS = {
    "youtube": "#E24B4A",
    "reddit":  "#378ADD",
}
TOPIC_COLORS = [
    "#534AB7", "#1D9E75", "#D85A30",
    "#185FA5", "#BA7517", "#993556", "#3C3489",
]


# ── 유틸 ──────────────────────────────────────────────────────────────────────
def detect_label_col(df: pd.DataFrame) -> str:
    for c in ("roberta_label", "vader_label", "label", "sentiment"):
        if c in df.columns:
            return c
    raise ValueError(f"감성 레이블 컬럼을 찾을 수 없습니다. 컬럼: {list(df.columns)}")


# ── 1. 감성 분석 시각화 ────────────────────────────────────────────────────────
def visualize_sentiment(platform: str):
    path = os.path.join(TABLES_DIR, f"sentiment_{platform}.csv")
    if not os.path.exists(path):
        print(f"  [건너뜀] 파일 없음: {path}")
        return

    df = pd.read_csv(path)
    label_col = detect_label_col(df)

    counts = df[label_col].str.lower().value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    counts["color"] = counts["sentiment"].map(SENTIMENT_COLORS).fillna("#B4B2A9")
    total = counts["count"].sum()
    counts["pct"] = (counts["count"] / total * 100).round(1)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "xy"}]],
        subplot_titles=("비율", "빈도"),
    )
    fig.add_trace(go.Pie(
        labels=counts["sentiment"],
        values=counts["count"],
        marker_colors=counts["color"],
        textinfo="label+percent",
        hole=0.45,
        hovertemplate="%{label}<br>%{value:,}개 (%{percent})<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=counts["sentiment"],
        y=counts["count"],
        marker_color=counts["color"],
        text=counts["pct"].astype(str) + "%",
        textposition="outside",
        hovertemplate="%{x}: %{y:,}개<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(
        title=dict(text=f"AI 음악 감성 분석 — {platform.upper()}  (총 {total:,}개)", font_size=18),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        height=460,
        margin=dict(t=80, b=40),
    )
    fig.update_yaxes(gridcolor="#f0f0f0", row=1, col=2)

    out = os.path.join(FIGURES_DIR, f"sentiment_{platform}.html")
    fig.write_html(out)
    print(f"  [완료] {out}")


# ── 2. LDA 시각화 ─────────────────────────────────────────────────────────────
def visualize_lda(platform: str):
    model_path = os.path.join(MODELS_DIR, f"lda_{platform}.model")
    dict_path  = os.path.join(MODELS_DIR, f"lda_{platform}.dict")

    for p in (model_path, dict_path):
        if not os.path.exists(p):
            print(f"  [건너뜀] 파일 없음: {p}")
            return

    lda_model  = LdaMulticore.load(model_path)
    dictionary = corpora.Dictionary.load(dict_path)

    # corpus를 전처리 CSV에서 재구성
    processed_path = os.path.join(PROCESSED_DIR, f"{platform}_cleaned.csv")
    if not os.path.exists(processed_path):
        print(f"  [건너뜀] 전처리 파일 없음 (corpus 재구성 불가): {processed_path}")
        return

    df = pd.read_csv(processed_path, dtype=str)
    text_col = next(
        (c for c in ("cleaned_text", "processed_text", "text", "comment", "body") if c in df.columns),
        None
    )
    if text_col is None:
        print(f"  [건너뜀] 텍스트 컬럼을 찾을 수 없음")
        return

    docs   = df[text_col].dropna().str.lower().str.split().tolist()
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    # pyLDAvis 인터랙티브 시각화
    prepared = gensimvis.prepare(lda_model, corpus, dictionary, mds="mmds")
    out_lda = os.path.join(FIGURES_DIR, f"lda_{platform}.html")
    pyLDAvis.save_html(prepared, out_lda)
    print(f"  [완료] {out_lda}")

    # 토픽별 상위 단어 바차트
    n_topics = lda_model.num_topics
    rows = (n_topics + 1) // 2
    fig = make_subplots(
        rows=rows, cols=2,
        subplot_titles=[f"Topic {i+1}" for i in range(n_topics)],
        vertical_spacing=0.12,
    )
    for i in range(n_topics):
        top_words = lda_model.show_topic(i, topn=10)
        words_  = [w for w, _ in top_words][::-1]
        scores  = [s for _, s in top_words][::-1]
        r, c = divmod(i, 2)
        fig.add_trace(go.Bar(
            x=scores, y=words_,
            orientation="h",
            marker_color=TOPIC_COLORS[i % len(TOPIC_COLORS)],
            showlegend=False,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ), row=r + 1, col=c + 1)

    fig.update_layout(
        title=dict(text=f"LDA 토픽별 핵심 단어 — {platform.upper()}", font_size=18),
        paper_bgcolor="white", plot_bgcolor="white",
        height=300 * rows,
        margin=dict(t=80, b=40),
    )
    fig.update_xaxes(gridcolor="#f0f0f0")

    out_words = os.path.join(FIGURES_DIR, f"lda_topwords_{platform}.html")
    fig.write_html(out_words)
    print(f"  [완료] {out_words}")


# ── 3. Word2Vec 3D 시각화 ─────────────────────────────────────────────────────
def visualize_w2v(platform: str, n_clusters: int = 5, max_words: int = 300):
    model_path = os.path.join(MODELS_DIR, f"w2v_{platform}.model")
    if not os.path.exists(model_path):
        print(f"  [건너뜀] 파일 없음: {model_path}")
        return

    model   = Word2Vec.load(model_path)
    wv      = model.wv
    words   = wv.index_to_key[:max_words]
    vectors = wv.vectors[:max_words]

    # PCA 3차원 축소
    pca     = PCA(n_components=3, random_state=42)
    reduced = pca.fit_transform(vectors)
    explained = pca.explained_variance_ratio_ * 100

    # KMeans 군집화
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)

    fig = go.Figure(go.Scatter3d(
        x=reduced[:, 0],
        y=reduced[:, 1],
        z=reduced[:, 2],
        mode="markers",
        marker=dict(
            size=5,
            color=labels,
            colorscale="Viridis",
            opacity=0.85,
            showscale=True,
            colorbar=dict(title="Cluster", thickness=12),
        ),
        text=words,
        hovertemplate="<b>%{text}</b><extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Word2Vec 3D 시각화 — {platform.upper()} (상위 {max_words}개 단어)", font_size=18),
        scene=dict(
            xaxis_title=f"PC1 ({explained[0]:.1f}%)",
            yaxis_title=f"PC2 ({explained[1]:.1f}%)",
            zaxis_title=f"PC3 ({explained[2]:.1f}%)",
            bgcolor="white",
        ),
        paper_bgcolor="white",
        height=700,
        margin=dict(t=80, b=20),
    )

    out = os.path.join(FIGURES_DIR, f"w2v_3d_{platform}.html")
    fig.write_html(out)
    print(f"  [완료] {out}")


# ── 4. 플랫폼 비교 시각화 ─────────────────────────────────────────────────────
def visualize_compare():
    dfs = {}
    for platform in ("youtube", "reddit"):
        path = os.path.join(TABLES_DIR, f"sentiment_{platform}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            label_col = detect_label_col(df)
            counts = df[label_col].str.lower().value_counts(normalize=True) * 100
            dfs[platform] = counts

    if len(dfs) < 2:
        print("  [건너뜀] 비교 시각화: YouTube/Reddit 결과 파일이 둘 다 필요합니다.")
        return

    sentiments = ["positive", "neutral", "negative"]
    fig = go.Figure()
    for platform, counts in dfs.items():
        fig.add_trace(go.Bar(
            name=platform.upper(),
            x=sentiments,
            y=[counts.get(s, 0) for s in sentiments],
            marker_color=PLATFORM_COLORS[platform],
            text=[f"{counts.get(s, 0):.1f}%" for s in sentiments],
            textposition="outside",
            hovertemplate=f"{platform.upper()} %{{x}}: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="YouTube vs Reddit — 감성 분포 비교", font_size=18),
        barmode="group",
        xaxis_title="감성",
        yaxis_title="비율 (%)",
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis=dict(gridcolor="#f0f0f0", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=500,
    )

    out = os.path.join(FIGURES_DIR, "sentiment_compare.html")
    fig.write_html(out)
    print(f"  [완료] {out}")


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AI 음악 감성 분석 시각화")
    parser.add_argument("--platform", choices=["youtube", "reddit", "all"], default="all")
    parser.add_argument("--n-clusters", type=int, default=5, help="Word2Vec KMeans 군집 수 (기본값: 5)")
    parser.add_argument("--max-words",  type=int, default=300, help="Word2Vec 시각화 단어 수 (기본값: 300)")
    args = parser.parse_args()

    platforms = ["youtube", "reddit"] if args.platform == "all" else [args.platform]

    print(f"\n{'='*50}")
    print(f"  AI Music Sentiment — 시각화 시작")
    print(f"  플랫폼: {', '.join(p.upper() for p in platforms)}")
    print(f"  출력 폴더: {FIGURES_DIR}")
    print(f"{'='*50}\n")

    for platform in platforms:
        print(f"── {platform.upper()} ──────────────────────────")
        visualize_sentiment(platform)
        visualize_lda(platform)
        visualize_w2v(platform, args.n_clusters, args.max_words)
        print()

    if len(platforms) == 2:
        print("── 비교 ────────────────────────────────────────")
        visualize_compare()

    print(f"\n{'='*50}")
    print("  모든 시각화 완료!")
    print(f"  결과 폴더: {FIGURES_DIR}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()