# AI Music Sentiment Analysis

AI 생성 음악에 대한 대중 인식을 분석하는 프로젝트  
YouTube 댓글과 Reddit 데이터를 수집하여 감정 분석, 토픽 모델링(LDA), Word2Vec을 수행합니다.

## Workflow

```mermaid
flowchart TD
    %% 1. Data Collection
    subgraph A [Data Collection]
        A1[youtube_search_ids.py] --> A2[youtube.py]
        A3[reddit_json.py]
    end

    A2 --> B[data/raw]
    A3 --> B

    %% 2. Processing
    subgraph C [Processing]
        B --> C1[Sentiment Mode]
        B --> C2[Topic Mode]
        B --> C3[DTM Mode]
    end

    %% 3. Processed Outputs
    subgraph D [Processed Data]
        C1 --> D1[processed/sentiment]
        C2 --> D2[processed/topic]
        C3 --> D3[processed/dtm]
    end

    %% 4. Analysis
    subgraph E [Analysis]
        D1 --> E1[sentiment.py]

        D2 --> E2[lda.py]
        D2 --> E3[word2vec.py]

        D3 --> E4[wordcloud_gen.py]
        D3 --> E5[top_words.py]
    end

    %% 5. Results
    subgraph F [Results]
        E1 --> F1[figures]
        E2 --> F1
        E3 --> F1
        E4 --> F1

        E1 --> F2[tables]
        E2 --> F2
        E3 --> F2
        E5 --> F2

        E2 --> F3[models]
        E3 --> F3
    end
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
