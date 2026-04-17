# AI Music Sentiment Analysis

AI 생성 음악에 대한 대중 인식을 분석하는 프로젝트  
YouTube 댓글과 Reddit 데이터를 수집하여 감정 분석, 토픽 모델링(LDA), Word2Vec을 수행합니다.

## Workflow

```mermaid
graph TD
    %% Collection Phase
    subgraph collection [collection/]
        A[youtube_search_ids.py] --> B[youtube.py]
        C[reddit_json.py]
    end

    B & C --> D[data/raw/]

    %% Processing Phase
    subgraph processing [processing/ — process.py]
        direction TB
        E{mode: sentiment}
        F{mode: topic}
        G{mode: dtm}
    end

    D --> E & F & G

    %% Processed Data
    E --> H([data/processed/sentiment/])
    F --> I([data/processed/topic/])
    G --> J([data/processed/dtm/])

    %% Analysis Phase
    subgraph analysis [analysis/]
        K[sentiment.py]
        L[lda.py]
        M[word2vec.py]
        N[wordcloud_gen.py]
        O[top_words.py]
    end

    H --> K
    I --> L & M
    J --> N & O

    %% Results Phase
    subgraph results [results/]
        P([figures/])
        Q([tables/])
        R([models/])
    end

    K & L & M & N --> P
    K & L & M & O --> Q
    L & M --> R
```

## Project Structure

```bash
.
├── data/                  # 데이터 저장소
│   ├── raw/               # 원본 데이터 (수집 결과)
│   ├── processed/         # 전처리된 데이터
│   └── collection_summary.json
│
├── src/                   # 소스 코드
│   ├── collection/        # 데이터 수집
│   │   ├── youtube.py
│   │   ├── reddit_json.py
│   │   └── youtube_search_ids.py
│   │
│   ├── processing/        # 데이터 전처리
│   │   ├── youtube.py
│   │   └── reddit.py
│   │
│   ├── analysis/          # 분석 로직
│   │   ├── sentiment.py
│   │   ├── lda.py
│   │   ├── word2vec.py
│   │   ├── wordcloud.py
│   │   └── top_words.py
│   │
│   └── main.py            # 전체 파이프라인 실행
│
├── notebooks/             # 실험용 노트북
├── results/               # 결과물
│   ├── figures/
│   ├── tables/
│   └── models/
│
├── LICENSE
└── README.md
```

## Data Collection

### YouTube

- YouTube API 사용
- 키워드 기반 영상 검색 후 댓글 수집

### Reddit

- Reddit comments json request 사용

## Data Processing

working

## License

This project is licensed under [BSD-3-Clause](LICENSE).
