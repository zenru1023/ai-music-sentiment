import argparse
import os
import re
import random

import matplotlib.pyplot as plt
import pandas as pd
from gensim import corpora, models
from gensim.models import CoherenceModel
from tqdm import tqdm

import pyLDAvis
import pyLDAvis.gensim_models

# ── 경로 설정 ────────────────────────────────────────────────────────────────

PLATFORM_PATHS = {
    "youtube": "data/processed/topic/youtube.csv",
    "reddit":  "data/processed/topic/reddit.csv",
}

OUTPUT_FIGURES = "results/figures/lda"
OUTPUT_MODELS  = "results/models/lda"
OUTPUT_TABLES  = "results/tables/lda"

# 폴더 자동 생성
for path in [OUTPUT_FIGURES, OUTPUT_MODELS, OUTPUT_TABLES]:
    os.makedirs(path, exist_ok=True)


# ── 데이터 로드 및 토큰화 ────────────────────────────────────────────────────

def load_and_tokenize(platform: str):
    path = PLATFORM_PATHS[platform]
    df = pd.read_csv(path, dtype=str)
    
    # 이미 process.py에서 불용어가 제거된 상태라면, 
    # 그냥 공백으로 잘라주기만 해도 충분합니다.
    def tokenize(text):
        # 혹시 모를 노이즈만 제거 (2글자 이상 단어만 추출)
        return re.findall(r"[a-z0-9]{2,}", str(text).lower())

    # 이미 전처리가 끝난 데이터이므로, 
    tokenized = df["text"].apply(tokenize).tolist()
    
    return df, [s for s in tokenized if len(s) >= 2]


# ── LDA 최적화 및 Coherence 계산 ─────────────────────────────────────────────

def compute_coherence_values(tokenized, start=4, limit=12, step=1):
    """
    토픽 수를 변화시키며 Coherence 점수를 계산하여 최적의 k를 탐색합니다.
    """
    dictionary = corpora.Dictionary(tokenized)
    # 빈도수가 너무 낮거나(5회 미만) 너무 높은(60% 이상) 단어 제거
    dictionary.filter_extremes(no_below=5, no_above=0.6)

    corpus = [dictionary.doc2bow(text) for text in tokenized]

    coherence_scores = []
    models_list = []

    print("\n[LDA Tuning] Coherence 탐색 시작...")

    # [수정] 무작위 샘플링을 통해 데이터 전반의 일관성 확인 (편향 방지)
    sample_size = min(30000, len(tokenized))
    coherence_samples = random.sample(tokenized, sample_size)

    for num_topics in tqdm(range(start, limit + 1, step), desc="LDA 모델 학습 중"):
        lda = models.LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=10,                      # 학습 횟수 증가로 정확도 향상
            workers=os.cpu_count() - 1,     # 사용 가능한 CPU 코어 활용
            random_state=42,
        )

        coherence_model = CoherenceModel(
            model=lda,
            texts=coherence_samples,        # 무작위 샘플 사용
            dictionary=dictionary,
            coherence="c_v",
        )

        score = coherence_model.get_coherence()
        coherence_scores.append(score)
        models_list.append(lda)

    return models_list, coherence_scores, dictionary, corpus


# ── 결과 저장 및 시각화 함수들 ───────────────────────────────────────────────

def plot_coherence(scores, start, limit, step, platform):
    x = list(range(start, limit + 1, step))
    plt.figure(figsize=(10, 6))
    plt.plot(x, scores, marker="o", color="#2c3e50", linestyle="-")
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score (c_v)")
    plt.title(f"Coherence Score by Topic Count — {platform.capitalize()}")
    plt.grid(True, alpha=0.3)

    path = os.path.join(OUTPUT_FIGURES, f"{platform}_coherence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"차트 저장 완료: {path}")

def save_ldavis(lda, corpus, dictionary, platform):
    print(f"[{platform}] pyLDAvis 생성 중...")
    vis = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary, sort_topics=False)
    path = os.path.join(OUTPUT_FIGURES, f"{platform}_ldavis.html")
    pyLDAvis.save_html(vis, path)
    print(f"HTML 시각화 완료: {path}")

def save_topics_table(lda, num_topics, platform):
    rows = []
    for i in range(num_topics):
        for rank, (word, weight) in enumerate(lda.show_topic(i, topn=15), 1):
            rows.append({
                "topic": i + 1,
                "rank": rank,
                "word": word,
                "weight": round(weight, 6),
            })
    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_TABLES, f"{platform}_topics.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"토픽 키워드 저장 완료: {path}")

def save_doc_topics(df, corpus, lda, platform):
    doc_topics = []
    for bow in corpus:
        topics = lda.get_document_topics(bow)
        if topics:
            dominant = max(topics, key=lambda x: x[1])
            doc_topics.append({"dominant_topic": dominant[0] + 1, "prob": round(dominant[1], 4)})
        else:
            doc_topics.append({"dominant_topic": -1, "prob": 0.0})

    topic_df = pd.DataFrame(doc_topics)
    result_df = pd.concat([df.reset_index(drop=True), topic_df], axis=1)
    path = os.path.join(OUTPUT_TABLES, f"{platform}_doc_topics.csv")
    result_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"문서별 토픽 매핑 저장 완료: {path}")

def print_best_topics(lda, num_topics):
    print("\n" + "=" * 60)
    print(f"최적 모델 요약 (Topics: {num_topics})")
    print("=" * 60)
    for i in range(num_topics):
        words = [w for w, _ in lda.show_topic(i, topn=8)]
        print(f"Topic {i+1:>2}: {', '.join(words)}")


# ── 메인 실행 루틴 ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LDA Topic Modeling")
    parser.add_argument("--platform", required=True, choices=["youtube", "reddit"])
    args = parser.parse_args()

    # 무작위 시드 고정 (재현성)
    random.seed(42)

    # 1. 데이터 로드
    df, tokenized = load_and_tokenize(args.platform)

    # 2. 최적 토픽 수 탐색 (4개에서 12개 사이)
    start_k, limit_k = 4, 12
    models_list, scores, dictionary, corpus = compute_coherence_values(
        tokenized, start=start_k, limit=limit_k
    )

    # 3. 최적 모델 선정 (Coherence 기준)
    best_idx = scores.index(max(scores))
    best_model = models_list[best_idx]
    best_k = best_idx + start_k

    print(f"\n최적 토픽 결정: {best_k}개 (Coherence: {scores[best_idx]:.4f})")

    # 4. 결과 시각화 및 파일 저장
    plot_coherence(scores, start_k, limit_k, 1, args.platform)
    print_best_topics(best_model, best_k)
    
    best_model.save(os.path.join(OUTPUT_MODELS, f"lda_{args.platform}.model"))
    dictionary.save(os.path.join(OUTPUT_MODELS, f"lda_{args.platform}.dict"))

    save_topics_table(best_model, best_k, args.platform)
    save_doc_topics(df, corpus, best_model, args.platform)
    save_ldavis(best_model, corpus, dictionary, args.platform)

    print("\n[LDA 분석 완료]")


if __name__ == "__main__":
    main()