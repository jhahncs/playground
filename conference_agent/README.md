# Conference Agent

학술 컨퍼런스 정보를 자동으로 수집하는 AI Agent입니다. WikiCFP, IEEE, ACM, Springer 등 다양한 소스에서 컨퍼런스 정보(마감일, 장소, 주제, CFP 등)를 스크래핑하여 CSV/Excel 형식으로 제공하고, 웹 대시보드를 통해 시각화합니다.

## Features

- 🌐 **다중 소스 지원**: WikiCFP, IEEE, ACM, Springer
- 🔍 **다중 분야 지원**: 머신러닝, AI, 컴퓨터 비전 등 다양한 학술 분야
- 📅 **주요 정보 수집**:
  - 논문 제출 마감일, 결과 통보일, 카메라 레디 마감일
  - 컨퍼런스 개최 장소, 날짜, 주제
  - Call for Papers (CFP) 내용 및 URL
- 🔮 **스마트 필터링**: 올해 및 향후 개최 예정 학회만 자동 필터링
- 🌟 **웹사이트 Enrichment**: 공식 사이트에서 추가 정보 수집
  - 기조연설자, 프로그램 위원회
  - 등록 정보, 제출 트랙
  - 소셜 미디어 링크
- ⚡ **효율적인 스크래핑**:
  - Rate limiting (WikiCFP 요구사항 준수)
  - 자동 재시도 및 캐싱
  - 진행상황 표시
- 📊 **데이터 품질 관리**:
  - 자동 데이터 검증
  - 품질 점수 계산 (0-1)
  - 중복 제거
- 💾 **다양한 출력 형식**:
  - CSV (데이터 분석용)
  - Excel (하이퍼링크, 조건부 서식 포함)
  - 통계 요약 보고서
- 🖥️ **웹 대시보드**:
  - 실시간 통계 및 시각화
  - 검색 및 필터링
  - 마감일 임박 알림
  - CSV/Excel 내보내기
- 📧 **이메일 알림**:
  - 마감일 임박 학회 자동 알림
  - Gmail, Outlook, Yahoo 등 지원
  - 커스터마이징 가능한 알림 기준
  - 일일/주간 자동 알림 (cron)

## Installation

1. Python 환경 설정:
```bash
cd /home/jhahn/playground/conference_agent
pip install -r requirements.txt
```

2. 디렉터리 구조 확인:
```
conference_agent/
├── data/
│   ├── cache/          # 캐시된 HTML
│   └── processed/      # 출력 CSV/Excel
├── scrapers/           # 스크래핑 모듈
├── parsers/            # HTML 파싱
├── models/             # 데이터 모델
├── storage/            # CSV/Excel 핸들러
├── utils/              # 유틸리티
├── config.py           # 설정
└── main.py             # 메인 프로그램
```

## Usage

### 기본 사용법

```bash
# 특정 카테고리 스크래핑 (WikiCFP)
python main.py --categories "machine learning,artificial intelligence"

# 여러 소스에서 최신 CFP 수집
python main.py --latest 50 --sources "wikicfp,ieee,acm,springer" --excel

# 키워드 검색
python main.py --search "deep learning" --limit 30

# 웹사이트 enrichment 활성화
python main.py --latest 30 --enrich

# 과거 학회 포함 (기본값: 올해 이후만)
python main.py --categories "computer vision" --include-past

# 기존 데이터 업데이트 (중복 제거)
python main.py --categories "computer vision" --update
```

### 웹 대시보드

웹 대시보드를 시작하려면:

```bash
python webapp.py
```

브라우저에서 http://localhost:5000 접속

대시보드 기능:
- 📊 실시간 통계 (총 학회 수, 마감일 임박 등)
- 🔍 검색 및 필터링 (소스, 연도, 품질 점수)
- 📅 마감일 임박 학회 보기
- 📥 CSV/Excel 내보내기

### 이메일 알림

이메일 알림 설정 (자세한 내용은 [EMAIL_SETUP.md](EMAIL_SETUP.md) 참고):

```bash
# 1. 환경 변수 설정
cp .env.example .env
# .env 파일에서 이메일 설정 입력

# 2. 이메일 설정 테스트
python main.py --test-email

# 3. 알림과 함께 스크래핑
python main.py --latest 50 --notify

# 4. 커스텀 알림 기간 (14일)
python main.py --latest 100 --notify --notify-days 14
```

**일일 자동 알림 설정 (cron)**:
```bash
# 매일 오전 9시에 7일 내 마감 학회 알림
0 9 * * * cd /home/jhahn/playground/conference_agent && python main.py --latest 100 --notify --update
```

### 고급 옵션

```bash
# 페이지 수 제한 및 품질 필터
python main.py --categories "NLP" --max-pages 3 --min-quality 0.7

# Excel 출력 with 서식
python main.py --latest 100 --excel --output data/processed/conferences_2026.xlsx

# 캐시 초기화 후 실행
python main.py --clear-cache --categories "cybersecurity"
```

## Configuration

`config.py`에서 다음을 설정할 수 있습니다:

```python
# 스크래핑 설정
RATE_LIMIT_SECONDS = 5.0  # WikiCFP rate limit
MAX_PAGES_PER_CATEGORY = 5
MAX_CONFERENCES_PER_RUN = 200

# 품질 필터
MIN_QUALITY_SCORE = 0.5
ENRICH_THRESHOLD = 0.7

# 카테고리 목록
CATEGORIES = [
    "machine learning",
    "artificial intelligence",
    "computer vision",
    # ...
]
```

## Output Format

### CSV Output
모든 컨퍼런스 정보가 표 형식으로 저장됩니다:
- `conferences.csv` - 메인 데이터
- `summary.txt` - 통계 요약

### Excel Output
포맷팅된 Excel 파일:
- **All Conferences**: 전체 데이터
- **Year XXXX**: 연도별 시트
- 하이퍼링크 (website_url, cfp_url)
- 조건부 서식:
  - 🔴 7일 이내 마감 (빨강)
  - 🟡 30일 이내 마감 (노랑)
  - ⚫ 마감 지난 것 (회색)

## Data Quality Score

각 컨퍼런스는 0-1 사이의 품질 점수를 가집니다:

- **0.5**: 필수 필드 (이름, 마감일, 개최일, 장소)
- **+0.1**: 결과 통보일
- **+0.1**: 카메라 레디 마감일
- **+0.2**: CFP 텍스트
- **+0.1**: 주제/카테고리
- **+0.05**: 종료일
- **+0.05**: 웹사이트 URL

## Architecture

```
┌─────────────────┐
│   main.py       │  ← CLI Entry Point
└────────┬────────┘
         │
    ┌────▼─────────────────┐
    │ ConferenceAgent      │
    │ (Orchestrator)       │
    └──┬────────┬─────┬────┘
       │        │     │
┌──────▼───┐ ┌─▼──┐ ┌▼──────┐
│WikiCFP   │ │CSV │ │Excel  │
│Scraper   │ │    │ │Handler│
└──┬───────┘ └────┘ └───────┘
   │
┌──▼──────────┐
│BaseScraper  │  ← Rate limiting, caching, retry
└──┬──────────┘
   │
┌──▼──────────┐
│WikiCFP      │  ← HTML parsing
│Parser       │
└─────────────┘
```

## Examples

### Example 1: 머신러닝 컨퍼런스 수집

```bash
python main.py --categories "machine learning" --max-pages 2 --excel
```

출력:
- `data/processed/conferences.csv`
- `data/processed/conferences.xlsx`
- `data/processed/summary.txt`

### Example 2: 최신 CFP 모니터링

```bash
python main.py --latest 100 --min-quality 0.7
```

### Example 3: 정기 업데이트 (cron)

```bash
# 매일 새로운 CFP를 기존 데이터에 추가
0 9 * * * cd /home/jhahn/playground/conference_agent && python main.py --latest 50 --update
```

## Data Model

```python
@dataclass
class Conference:
    conference_id: str
    name: str
    acronym: Optional[str]
    year: int
    location: str
    is_virtual: bool
    is_hybrid: bool
    conference_start: date
    conference_end: date
    submission_deadline: date
    notification_date: date
    camera_ready_deadline: date
    topics: List[str]
    categories: List[str]
    cfp_text: str
    cfp_url: str
    website_url: str
    data_quality_score: float
```

## Troubleshooting

### Rate Limiting 오류
WikiCFP에서 429 에러가 발생하면:
- `config.py`에서 `RATE_LIMIT_SECONDS` 증가 (기본 5초)
- `--clear-cache` 옵션으로 캐시 초기화

### 파싱 오류
HTML 구조가 변경된 경우:
- `parsers/wikicfp_parser.py` 업데이트 필요
- GitHub issues에 보고

### 메모리 부족
대량 스크래핑 시:
- `--limit` 옵션으로 제한
- `--max-pages` 줄이기

## Recent Updates (2026-02-13)

- ✅ **올해 이후 학회 필터링**: `--include-past` 플래그로 제어 (기본값: 올해 이후만)
- ✅ **웹사이트 enrichment**: `--enrich` 플래그로 공식 사이트에서 추가 정보 수집
- ✅ **다중 소스 지원**: IEEE, ACM, Springer 스크래퍼 추가 (`--sources` 옵션)
- ✅ **웹 대시보드**: Flask 기반 대시보드 with 검색, 필터, 통계
- ✅ **이메일 알림**: 마감일 임박 학회 자동 이메일 알림 (`--notify` 옵션)

## Future Enhancements

- [ ] 데이터베이스 백엔드 (SQLite)
- [ ] Google Calendar 연동
- [ ] API 엔드포인트 (RESTful API)
- [ ] 사용자 계정 및 즐겨찾기
- [ ] 모바일 앱 (React Native)

## License

MIT License

## Author

Conference Agent
Created: 2026-02-13
