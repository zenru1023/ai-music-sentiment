# AI Music Sentiment Analysis

AI 생성 음악에 대한 대중 인식을 분석하는 프로젝트  
YouTube 댓글과 Reddit 데이터를 수집하여 감정 분석, 토픽 모델링(LDA), Word2Vec을 수행합니다.

## Workflow

```mermaid
graph TD
    %% Collection Stage
    subgraph collection ["collection/"]
        A[youtube_search_ids.py<br/>검색 결과 -> video_ids.txt] --> B[youtube.py<br/>order=time, 최대 10,000건]
        C[reddit_json.py<br/>public JSON, 6개 갤러리]
    end

    %% Raw Data
    B --> D([data/raw/<br/>yt_*.csv · rd_*.csv])
    C --> D

    %% Processing Stage
    subgraph processing ["processing/ — process.py --platform youtube/reddit --mode ..."]
        E{{--mode sentiment<br/>감성 분석 · 정제}}
        F{{--mode topic<br/>토픽 모델링 · 정제}}
        G{{--mode dtm<br/>단어 행렬 생성 및 정제}}
    end

    D --> E
    D --> F
    D --> G

    %% Processed Data
    subgraph processed_data ["data/processed/"]
        H([sentiment/youtube-reddit.csv])
        I([topic/youtube-reddit.csv])
        J([dtm/youtube-reddit.csv])
    end

    E --> H
    F --> I
    G --> J

    %% Analysis Stage
    subgraph analysis ["analysis/ — --platform youtube/reddit"]
        K[sentiment.py<br/>RoBERTa · GPU]
        L[lda.py<br/>LdaMulticore]
        M[word2vec.py<br/>Skip-gram · 3D PCA]
        N[wordcloud_gen.py]
        O[top_words.py]
    end

    H --> K
    I --> L
    I --> M
    J --> N
    J --> O

    %% Results Stage
    subgraph results ["results/"]
        P([figures/<br/>sentiment · lda · w2v · wordcloud])
        Q([tables/<br/>sentiment · lda · top_words · w2v])
        R([models/<br/>lda · wordvec])
    end

    K --> P
    L --> P
    M --> P
    N --> P

    K --> Q
    L --> Q
    O --> Q
    M --> Q

    L --> R
    M --> R
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
