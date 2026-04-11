import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── 경로 설정 ────────────────────────────────────────────────────────────────
PLATFORM_PATHS = {
    "youtube": "data/processed/topic/youtube.csv",
    "reddit":  "data/processed/topic/reddit.csv",
}
OUTPUT_MODELS  = "results/models/word2vec"
OUTPUT_TABLES  = "results/tables/word2vec"
OUTPUT_FIGURES = "results/figures/word2vec"

for path in [OUTPUT_MODELS, OUTPUT_TABLES, OUTPUT_FIGURES]:
    os.makedirs(path, exist_ok=True)

# ── 분석 대상 핵심 단어 ──────────────────────────────────────────────────────
SEED_WORDS = [
    "ai", "suno", "udio", "generated", "artificial",
    "creativity", "human", "copyright", "emotion",
    "quality", "replace", "artist", "original",
    "industry", "future", "authentic", "tool",
]

# ── 데이터 로드 및 토큰화 ────────────────────────────────────────────────────
def load_and_tokenize(platform: str):
    path = PLATFORM_PATHS[platform]
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일 없음: {path}")
    
    df = pd.read_csv(path, dtype=str)
    df = df[df["text"].notna() & (df["text"].str.strip() != "")].copy()
    
    def tokenize(text: str):
        # 영어 데이터 특화: 2글자 이상 (ai 보존)
        return re.findall(r"[a-z0-9]{2,}", str(text).lower())

    sentences = df["text"].apply(tokenize).tolist()
    sentences = [s for s in sentences if len(s) >= 2]
    return df, sentences

# ── 유사어 분석 및 콘솔 출력 (기존 기능 복구) ──────────────────────────────────
def analyze_and_print_similar(model, platform):
    print("\n" + "=" * 60)
    print("핵심 단어(Seed Words) 기반 유사어 분석")
    print("=" * 60)
    
    rows = []
    available = [w for w in SEED_WORDS if w in model.wv]
    
    for word in available:
        similars = model.wv.most_similar(word, topn=10)
        print(f"\n'{word}'와(과) 가장 유사한 단어:")
        for rank, (sim_word, score) in enumerate(similars, 1):
            print(f"   {rank:>2}. {sim_word:<18} {score:.4f}")
            rows.append({
                "seed_word": word,
                "rank": rank,
                "similar_word": sim_word,
                "similarity": round(score, 4),
            })
            
    # CSV 저장
    out_path = os.path.join(OUTPUT_TABLES, f"w2v_{platform}_similar.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n분석 결과 CSV 저장 → {out_path}")

# ── 3D 클러스터 시각화 (참고 코드 기능 이식) ──────────────────────────────────
def save_3d_interactive_map(model, platform, n_words=300):
    words = list(model.wv.index_to_key[:n_words])
    vectors = model.wv.vectors[:n_words]

    # PCA 차원 축소 및 KMeans 군집화
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(vectors)
    kmeans = KMeans(n_clusters=6, random_state=42)
    clusters = kmeans.fit_predict(vectors)

    fig = go.Figure(data=[go.Scatter3d(
        x=reduced[:, 0], y=reduced[:, 1], z=reduced[:, 2],
        mode='markers+text',
        marker=dict(size=4, color=clusters, colorscale='Viridis', opacity=0.7),
        text=words,
        hovertemplate='<b>%{text}</b><br>Cluster: %{marker.color}'
    )])

    fig.update_layout(title=f"3D Word Map - {platform.upper()}", margin=dict(l=0,r=0,b=0,t=40))
    html_path = os.path.join(OUTPUT_FIGURES, f"w2v_{platform}_3d_map.html")
    fig.write_html(html_path)
    print(f"3D HTML 시각화 저장 → {html_path}")

# ── 메인 실행 ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=["youtube", "reddit"], required=True)
    args = parser.parse_args()

    print(f"\n[Word2Vec 분석 시작] 대상: {args.platform}")
    _, sentences = load_and_tokenize(args.platform)

    # 모델 학습
    model = Word2Vec(
        sentences=sentences,
        vector_size=200,
        window=5,
        min_count=5,
        workers=4,
        sg=1,
        epochs=15,
        seed=42
    )
    print(f"최종 어휘 크기: {len(model.wv):,}개")

    # 1. 기존처럼 화면에 유사어 쫙 뽑아주기
    analyze_and_print_similar(model, args.platform)

    # 2. 참고 코드의 3D 시각화 수행
    save_3d_interactive_map(model, args.platform)

    # 3. 모델 저장
    model_path = os.path.join(OUTPUT_MODELS, f"w2v_{args.platform}.model")
    model.save(model_path)
    print(f"모델 파일 저장 완료 → {model_path}")

if __name__ == "__main__":
    main()