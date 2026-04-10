# analysis

## 소스 목록

| Name      | Description          |
| --------- | -------------------- |
| wordcloud | 워드 클라우드 생성   |
| top_words | 상위 20 단어         |
| sentiment | VADER + Transformers |
| lda       | LDA 토픽 모델링      |
| word2vec  | Word2Vec             |

##

```bash
# 1. 워드클라우드 & 상위 단어 (빠름)
uv run src/analysis/wordcloud.py --platform youtube
uv run src/analysis/wordcloud.py --platform reddit

uv run src/analysis/top_words.py --platform youtube
uv run src/analysis/top_words.py --platform reddit

# 2. 감정 분석 (Transformers는 처음 실행 시 모델 다운로드 ~1GB)
uv run src/analysis/sentiment.py --platform youtube
uv run src/analysis/sentiment.py --platform reddit

# 3. LDA & Word2Vec (고사양 컴퓨터에서 돌리길 권장)
uv run src/analysis/lda.py --platform youtube --num-topics 7
uv run src/analysis/lda.py --platform reddit --num-topics 7

uv run src/analysis/word2vec.py --platform youtube
uv run src/analysis/word2vec.py --platform reddit
```

## 결과물 저장 위치

- results/
  - figures/
  - tables/
  - models/
