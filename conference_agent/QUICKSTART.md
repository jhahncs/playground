# Quick Start Guide

## 설치

```bash
cd /home/jhahn/playground/conference_agent
pip install -r requirements.txt
```

## 기본 사용법

### 1. 올해 이후 학회만 스크래핑 (기본 동작)

```bash
# WikiCFP에서 머신러닝 학회 수집
python main.py --categories "machine learning" --max-pages 2

# 과거 학회도 포함하려면
python main.py --categories "machine learning" --include-past
```

### 2. 여러 소스에서 데이터 수집

```bash
# WikiCFP, IEEE, ACM, Springer에서 최신 50개 학회 수집
python main.py --latest 50 --sources "wikicfp,ieee,acm,springer" --excel
```

### 3. 웹사이트 Enrichment

```bash
# 공식 웹사이트에서 추가 정보 수집 (기조연설자, 프로그램 위원회 등)
python main.py --latest 20 --enrich
```

### 4. 웹 대시보드 실행

```bash
# 먼저 데이터 수집
python main.py --latest 100 --sources "wikicfp,ieee,acm" --excel

# 웹 대시보드 시작
python webapp.py
```

브라우저에서 http://localhost:5000 접속

## 주요 명령어 옵션

### 스크래핑 모드
- `--categories "topic1,topic2"`: 특정 카테고리 스크래핑
- `--latest N`: 최신 N개 CFP 수집
- `--search "keyword"`: 키워드로 검색

### 소스 선택
- `--sources "wikicfp,ieee,acm,springer"`: 사용할 소스 지정 (기본: wikicfp)

### 필터링
- `--include-past`: 과거 학회도 포함 (기본: 올해 이후만)
- `--min-quality 0.7`: 최소 품질 점수 (0-1)

### 기능
- `--enrich`: 웹사이트 enrichment 활성화
- `--excel`: Excel 출력 생성
- `--update`: 기존 데이터에 병합 (중복 제거)
- `--clear-cache`: 캐시 초기화

## 출력 파일

- `data/processed/conferences.csv`: 메인 데이터
- `data/processed/conferences.xlsx`: Excel 파일 (--excel 옵션 사용 시)
- `data/processed/summary.txt`: 통계 요약
- `conference_agent.log`: 로그 파일

## 웹 대시보드 기능

1. **통계 대시보드**: 총 학회 수, 마감일 임박 학회 등
2. **검색**: 학회명, 주제, 카테고리, 위치로 검색
3. **필터링**: 소스, 연도, 품질 점수로 필터
4. **마감일 임박 보기**: 7일/30일 내 마감 학회 확인
5. **내보내기**: CSV/Excel 다운로드

## 예제

### 예제 1: 빠른 시작

```bash
# 최신 30개 학회 수집 및 대시보드 시작
python main.py --latest 30 --excel
python webapp.py
```

### 예제 2: 고품질 데이터 수집

```bash
# 여러 소스에서 고품질 데이터만 수집
python main.py --latest 100 \
  --sources "wikicfp,ieee,acm,springer" \
  --min-quality 0.7 \
  --enrich \
  --excel
```

### 예제 3: 특정 분야 집중 수집

```bash
# 머신러닝, AI, 컴퓨터 비전 학회 수집
python main.py \
  --categories "machine learning,artificial intelligence,computer vision" \
  --max-pages 3 \
  --excel
```

## 문제 해결

### Rate Limiting 오류
```bash
# 캐시 초기화 후 재시도
python main.py --clear-cache --latest 50
```

### 데이터가 없을 때
```bash
# 먼저 데이터를 수집해야 웹 대시보드가 작동합니다
python main.py --latest 50 --excel
```

## 정기 실행 (Cron)

```bash
# 매일 오전 9시에 새로운 CFP 수집
0 9 * * * cd /home/jhahn/playground/conference_agent && python main.py --latest 50 --update
```

## 새로운 기능 (2026-02-13)

1. ✅ **올해 이후 학회만 자동 필터링** - 기본적으로 과거 학회는 제외
2. ✅ **다중 소스 지원** - WikiCFP, IEEE, ACM, Springer
3. ✅ **웹사이트 Enrichment** - 공식 사이트에서 추가 정보 자동 수집
4. ✅ **웹 대시보드** - 실시간 검색, 필터링, 통계 시각화
5. ✅ **이메일 알림** - 마감일 임박 학회 자동 알림

## 이메일 알림 설정

### 빠른 설정

```bash
# 1. 환경 파일 복사
cp .env.example .env

# 2. .env 파일 수정 (이메일 설정 입력)
nano .env

# 3. 이메일 설정 테스트
python main.py --test-email

# 4. 알림과 함께 실행
python main.py --latest 50 --notify
```

### Gmail 설정 (가장 쉬움)

1. [Google App Passwords](https://myaccount.google.com/apppasswords) 에서 앱 비밀번호 생성
2. `.env` 파일 수정:
```bash
EMAIL_ENABLED=true
SENDER_EMAIL=your-gmail@gmail.com
SENDER_PASSWORD=xxxx xxxx xxxx xxxx  # 16자리 앱 비밀번호
RECIPIENT_EMAIL=your-gmail@gmail.com
```

### 이메일 알림 예제

```bash
# 7일 내 마감 학회 알림
python main.py --latest 50 --notify

# 14일 내 마감 학회 알림
python main.py --latest 100 --notify --notify-days 14

# 매일 자동 알림 (cron)
0 9 * * * cd /home/jhahn/playground/conference_agent && python main.py --latest 100 --notify --update
```

자세한 내용은 [EMAIL_SETUP.md](EMAIL_SETUP.md) 참고
