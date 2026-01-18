import pandas as pd
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import time
import re
import logging
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os
import urllib.request
from konlpy.tag import Okt
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import csv  # CSV 저장 옵션 설정을 위해 추가

# ==========================================
# 0. 로거 설정
# ==========================================
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [Line %(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)


# ==========================================
# 1. 함수 정의 구간
# ==========================================
def format_rfc3339(date_str, is_end=False):
    """
    YYYY-MM-DD 형식의 날짜 문자열을 YouTube API가 요구하는 RFC 3339 포맷으로 변환합니다.
    """
    if not date_str:
        return None
    
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        logger.warning(f"날짜 형식이 올바르지 않습니다 ({date_str}). YYYY-MM-DD 형식을 사용해주세요.")
        return None

    if is_end:
        return f"{date_str}T23:59:59Z"
    else:
        return f"{date_str}T00:00:00Z"

def get_font_path():
    """
    OS에 맞는 한글 폰트 경로를 찾아서 반환합니다. 없으면 다운로드합니다.
    """
    font_path = None
    system_name = platform.system()
    font_candidates = []

    if system_name == 'Windows':
        font_candidates = ["c:/Windows/Fonts/malgun.ttf", "c:/Windows/Fonts/gulim.ttc"]
    elif system_name == 'Darwin': # Mac
        font_candidates = ["/System/Library/Fonts/AppleSDGothicNeo.ttc", "/Library/Fonts/AppleGothic.ttf"]
    else: # Linux
        font_candidates = ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"]

    for candidate in font_candidates:
        if os.path.exists(candidate):
            font_path = candidate
            break
    
    if font_path is None:
        current_dir = os.getcwd()
        local_font_path = os.path.join(current_dir, "NanumGothic.ttf")
        
        if not os.path.exists(local_font_path):
            url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
            try:
                urllib.request.urlretrieve(url, local_font_path)
                font_path = local_font_path
            except Exception:
                pass
        else:
            font_path = local_font_path
            
    return font_path

def load_stopwords(file_path):
    """
    불용어 파일을 읽어서 set 형태로 반환합니다.
    파일이 없거나 읽을 수 없는 경우 빈 set을 반환합니다.
    """
    stopwords = set()
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        stopwords.add(word)
            logger.info(f"불용어 파일 로드 완료: {len(stopwords)}개 단어")
        except Exception as e:
            logger.error(f"불용어 파일 읽기 실패: {e}")
    elif file_path:
        logger.warning(f"불용어 파일을 찾을 수 없습니다: {file_path}")
    
    return stopwords

def search_videos(youtube, query, max_results=5, start_date=None, end_date=None):
    """
    키워드와 기간으로 영상을 검색하고, 상세 정보를 반환합니다.
    query가 리스트일 경우 OR 연산(|)으로 결합하여 검색합니다.
    max_results가 50개를 초과할 경우 페이지네이션을 통해 추가 수집합니다.
    """
    videos_data = []
    video_ids = []
    
    # 리스트 형태의 쿼리를 OR 조건 문자열로 변환
    if isinstance(query, list):
        query_str = "|".join(query)
    else:
        query_str = query
        
    try:
        # 날짜 필터 변환
        published_after = format_rfc3339(start_date)
        published_before = format_rfc3339(end_date, is_end=True)
        
        if published_after:
            logger.info(f"검색 시작일 설정: {start_date}")
        if published_before:
            logger.info(f"검색 종료일 설정: {end_date}")

        # 페이지네이션을 통한 50개 이상 영상 수집 로직 추가
        next_page_token = None
        
        while len(video_ids) < max_results:
            # 이번 요청에서 가져올 개수 계산 (API 최대 50개 제한)
            remaining_count = max_results - len(video_ids)
            request_limit = min(remaining_count, 50)
            
            search_params = {
                'q': query_str,
                'type': "video",
                'part': "id",
                'maxResults': request_limit,
                'order': "relevance"
            }
            
            if next_page_token:
                search_params['pageToken'] = next_page_token

            if published_after:
                search_params['publishedAfter'] = published_after
            if published_before:
                search_params['publishedBefore'] = published_before
            
            time.sleep(0.3)
            search_request = youtube.search().list(**search_params)
            search_response = search_request.execute()
            
            items = search_response.get('items', [])
            if not items:
                break
                
            for item in items:
                if 'videoId' in item['id']:
                    vid = item['id']['videoId']
                    if vid not in video_ids:
                        video_ids.append(vid)
            
            # 다음 페이지 토큰 확인
            next_page_token = search_response.get('nextPageToken')
            if not next_page_token:
                break
        
        # 수집된 ID가 없으면 종료
        if not video_ids:
            return []
            
        # 정확히 max_results만큼 자르기
        video_ids = video_ids[:max_results]

        # 2. 비디오 상세 정보 API 호출 (조회수, 채널명 등 확보)
        # videos().list는 id 파라미터에 최대 50개까지만 허용되므로 청크로 나누어 요청
        for i in range(0, len(video_ids), 50):
            chunk_ids = video_ids[i:i+50]
            
            video_request = youtube.videos().list(
                part="snippet,statistics",
                id=','.join(chunk_ids)
            )
            video_response = video_request.execute()

            for item in video_response['items']:
                snippet = item['snippet']
                statistics = item['statistics']
                
                video_info = {
                    'id': item['id'],
                    'title': snippet['title'],
                    'channel': snippet['channelTitle'],
                    'published_at': snippet['publishedAt'][:10],
                    'view_count': statistics.get('viewCount', 0),
                    'like_count': statistics.get('likeCount', 0),
                    'comment_count': statistics.get('commentCount', 0)
                }
                videos_data.append(video_info)
            
    except Exception as e:
        logger.error(f"검색 중 에러 발생: {e}")
        
    return videos_data

def get_video_script(video_id):
    """
    youtube_transcript_api를 사용하여 영상의 자막을 텍스트로 가져옵니다.
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
        formatter = TextFormatter()
        script_text = formatter.format_transcript(transcript)
        return script_text
    except Exception:
        return None

def get_comments(youtube, video_info, max_comments_per_video=None):
    """
    YouTube API를 사용하여 해당 영상의 댓글을 수집합니다.
    """
    comments = []
    video_id = video_info['id']
    video_title = video_info['title']
    
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText",
            order="time"
        )
        
        limit_msg = f"(최대 {max_comments_per_video}개)" if max_comments_per_video else "(제한 없음)"
        logger.info(f" -> 댓글 수집 시작: {video_title[:30]}... {limit_msg}")

        while request:
            # API 호출 후 잠시 대기 (차단 방지)
            time.sleep(0.3)
            
            response = request.execute()

            for item in response['items']:
                if max_comments_per_video and len(comments) >= max_comments_per_video:
                    break

                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    '영상제목': video_title,
                    '채널명': video_info['channel'],
                    '영상조회수': video_info['view_count'],
                    '영상게시일': video_info['published_at'],
                    'Video_ID': video_id,
                    '작성자': comment['authorDisplayName'],
                    '내용': comment['textDisplay'],
                    '날짜': comment['publishedAt'],
                    '좋아요수': comment['likeCount']
                })

            if len(comments) % 100 == 0 and len(comments) > 0:
                logger.info(f"    현재 {len(comments)}개 수집 중...")

            if max_comments_per_video and len(comments) >= max_comments_per_video:
                logger.info(f"    목표 수량 도달로 수집 중단 ({len(comments)}개)")
                break

            if 'nextPageToken' in response:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    textFormat="plainText",
                    order="time",
                    pageToken=response['nextPageToken']
                )
            else:
                break
                
    except Exception as e:
        if "commentsDisabled" in str(e):
            logger.warning("    (댓글 사용 중지된 영상입니다)")
        else:
            logger.error(f"    에러 발생: {e}")
        return []

    return comments

def generate_statistics(df, num_of_total_videos, output_base_name):
    """
    수집된 댓글 데이터프레임에 대한 기본 통계를 계산하고 파일로 저장합니다.
    """
    logger.info("기본 통계 분석을 시작합니다...")
    
    stats_lines = []
    stats_lines.append("==========================================")
    stats_lines.append(" [ 수집 데이터 기본 통계 리포트 ]")
    stats_lines.append("==========================================\n")

    unique_videos = df[['Video_ID', '영상제목', '채널명', '영상게시일']].drop_duplicates().copy()
    video_count = len(unique_videos)
    stats_lines.append(f"1. 총 수집 영상 개수: {num_of_total_videos}개\n")
    stats_lines.append(f"   - 댓글이 없는 영상 개수: {num_of_total_videos - video_count}개\n")
    stats_lines.append(f"   - 댓글이 있는 영상 개수: {video_count}개\n")

    comments_per_video = df.groupby('Video_ID').size()
    stats_lines.append("2. 영상 별 수집 댓글 수 통계")
    stats_lines.append(f"   - 총 댓글 수: {len(df)}개")
    stats_lines.append(f"   - 평균 댓글 수: {comments_per_video.mean():.2f}개")
    stats_lines.append(f"   - 최대 댓글 수: {comments_per_video.max()}개")
    stats_lines.append(f"   - 최소 댓글 수: {comments_per_video.min()}개\n")

    df['word_count'] = df['내용'].astype(str).apply(lambda x: len(x.split()))
    stats_lines.append("3. 댓글 단어 수 통계")
    stats_lines.append(f"   - 평균 단어 수: {df['word_count'].mean():.2f}개")
    stats_lines.append(f"   - 최대 단어 수: {df['word_count'].max()}개")
    stats_lines.append(f"   - 최소 단어 수: {df['word_count'].min()}개\n")

    unique_videos['month'] = unique_videos['영상게시일'].astype(str).str[:7]
    videos_by_month = unique_videos['month'].value_counts().sort_index()
    
    stats_lines.append("4. 월별 영상 개수 (게시일 기준)")
    for month, count in videos_by_month.items():
        stats_lines.append(f"   - {month}: {count}개")
    stats_lines.append("")

    videos_by_channel = unique_videos['채널명'].value_counts()
    stats_lines.append("5. 채널별 영상 개수")
    for channel, count in videos_by_channel.items():
        stats_lines.append(f"   - {channel}: {count}개")
    stats_lines.append("")

    stats_file = f"{output_base_name}_statistics.txt"
    try:
        with open(stats_file, "w", encoding="utf-8") as f:
            f.write("\n".join(stats_lines))
        logger.info(f"통계 리포트가 '{stats_file}'로 저장되었습니다.")
        
        if 'display' in globals():
            print("\n" + "\n".join(stats_lines))
            
    except Exception as e:
        logger.error(f"통계 파일 저장 중 오류 발생: {e}")

def generate_wordcloud(text_data, output_base_name, data_type="comment", stopwords=None, folder_name=None):
    """
    입력받은 텍스트 리스트(text_data)를 형태소 분석하고 워드 클라우드를 생성합니다.
    (TF-IDF 기준과 빈도수 기준 2가지 버전을 생성)
    불용어(stopwords)가 포함된 경우 분석에서 제외합니다.
    folder_name이 있으면 제목에 추가합니다.
    """
    logger.info(f"[{data_type}] 형태소 분석 및 TF-IDF/빈도수 분석을 시작합니다...")
    
    if stopwords is None:
        stopwords = set()

    # 제목 접미사 설정
    title_suffix = f" - {folder_name}" if folder_name else ""

    raw_texts = []
    # 데이터 타입에 따라 텍스트 리스트 추출
    if isinstance(text_data, pd.Series):
        raw_texts = text_data.dropna().astype(str).tolist()
    elif isinstance(text_data, list):
        if text_data and isinstance(text_data[0], dict):
             # {'text': ..., 'date': ...} 형태인 경우
             raw_texts = [t['text'] for t in text_data if t.get('text')]
        else:
             raw_texts = [str(t) for t in text_data if t]
    
    if not raw_texts:
        logger.warning(f"[{data_type}] 워드 클라우드를 생성할 텍스트 데이터가 없습니다.")
        return

    try:
        okt = Okt()
        preprocessed_docs = []

        for idx, text in enumerate(raw_texts):
            if (idx + 1) % 500 == 0:
                logger.info(f"    [{data_type}] 형태소 분석 진행 중... ({idx + 1}/{len(raw_texts)})")
            
            tagged_words = okt.pos(text, stem=True)
            # 불용어 필터링 추가
            tokens = [word for word, tag in tagged_words 
                      if tag in ['Noun', 'Adjective', 'Verb'] and word not in stopwords]
            
            if len(tokens) >= 2:
                preprocessed_docs.append(" ".join(tokens))

        if not preprocessed_docs:
            logger.warning(f"[{data_type}] 분석 결과 유의미한 텍스트 데이터를 확보하지 못했습니다.")
            return

        # 2. TF-IDF 및 빈도수 계산
        logger.info(f"[{data_type}] TF-IDF 및 빈도수 계산 중...")
        token_pattern = r"(?u)\b\w+\b"
        ngram_range = (2, 2)
        
        count_vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=1, token_pattern=token_pattern)
        count_matrix = count_vectorizer.fit_transform(preprocessed_docs)
        
        tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=1, token_pattern=token_pattern)
        tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs)
        
        words = count_vectorizer.get_feature_names_out()
        sum_counts = np.array(count_matrix.sum(axis=0)).flatten()
        sum_tfidf = np.array(tfidf_matrix.sum(axis=0)).flatten()
        
        result_df = pd.DataFrame({
            '단어': words,
            '빈도수': sum_counts,
            'TF-IDF Score': sum_tfidf
        })
        
        # TF-IDF 기준 상위 100개
        top_100_tfidf = result_df.sort_values(by='TF-IDF Score', ascending=False).head(100)
        
        # 3. CSV 저장 (워드클라우드 데이터)
        wc_data_file = f"{output_base_name}_wordcloud_data_{data_type}.csv"
        # [수정] QUOTE_ALL로 통일하여 호환성 확보, escapechar 추가
        top_100_tfidf.to_csv(wc_data_file, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL, escapechar='\\')
        logger.info(f"[{data_type}] 분석 데이터가 '{wc_data_file}'로 저장되었습니다.")
        
        # 4. 워드 클라우드 생성 (2종류: TF-IDF, Frequency)
        font_path = get_font_path()
        wc_kwargs = {
            'background_color': 'white',
            'width': 800,
            'height': 600,
            'max_words': 100,
            'font_path': font_path
        }

        # 폰트 속성 생성 (제목용)
        font_prop = None
        if font_path:
            font_prop = fm.FontProperties(fname=font_path)

        # (1) TF-IDF 기준 워드클라우드
        word_dict_tfidf = dict(zip(top_100_tfidf['단어'], top_100_tfidf['TF-IDF Score']))
        wc_tfidf = WordCloud(**wc_kwargs).generate_from_frequencies(word_dict_tfidf)

        plt.figure(figsize=(10, 8))
        plt.imshow(wc_tfidf, interpolation='bilinear')
        plt.axis('off')
        
        # 제목 추가 (TF-IDF)
        title_tfidf = f"[{data_type}] WordCloud (TF-IDF){title_suffix}"
        if font_prop:
            plt.title(title_tfidf, fontsize=20, fontproperties=font_prop)
        else:
            plt.title(title_tfidf, fontsize=20)
            
        file_name_tfidf = f"{output_base_name}_wordcloud_{data_type}_tfidf.png"
        plt.savefig(file_name_tfidf)
        logger.info(f"[{data_type}] TF-IDF 워드 클라우드 이미지가 저장되었습니다: {file_name_tfidf}")
        plt.close()

        # (2) 빈도수(Frequency) 기준 워드클라우드
        # 빈도수 기준 상위 100개 다시 정렬
        top_100_freq = result_df.sort_values(by='빈도수', ascending=False).head(100)
        word_dict_freq = dict(zip(top_100_freq['단어'], top_100_freq['빈도수']))
        wc_freq = WordCloud(**wc_kwargs).generate_from_frequencies(word_dict_freq)

        plt.figure(figsize=(10, 8))
        plt.imshow(wc_freq, interpolation='bilinear')
        plt.axis('off')

        # 제목 추가 (Frequency)
        title_freq = f"[{data_type}] WordCloud (Frequency){title_suffix}"
        if font_prop:
            plt.title(title_freq, fontsize=20, fontproperties=font_prop)
        else:
            plt.title(title_freq, fontsize=20)

        file_name_freq = f"{output_base_name}_wordcloud_{data_type}_freq.png"
        plt.savefig(file_name_freq)
        logger.info(f"[{data_type}] 빈도수 워드 클라우드 이미지가 저장되었습니다: {file_name_freq}")
        plt.close()

    except Exception as e:
        logger.error(f"[{data_type}] 워드 클라우드 생성 중 오류 발생: {e}")

def generate_monthly_trend(data, output_base_name, data_type="comment", stopwords=None, folder_name=None):
    """
    월별 키워드(상위 10개 2-gram) 빈도수 변화를 꺾은선 그래프로 그립니다.
    빈도수 기준과 TF-IDF 기준 각각 상위 10개를 추출하여 그래프를 생성합니다.
    불용어(stopwords)가 포함된 경우 분석에서 제외합니다.
    folder_name이 있으면 제목에 추가합니다.
    """
    logger.info(f"[{data_type}] 월별 키워드 트렌드 분석을 시작합니다 (Freq & TF-IDF, 2-gram)...")
    
    if stopwords is None:
        stopwords = set()

    # 제목 접미사 설정
    title_suffix = f" - {folder_name}" if folder_name else ""

    # 1. 데이터 표준화 (date, text 리스트로 변환)
    standardized_data = [] 
    
    if (data_type == "comment" or data_type == "title") and isinstance(data, pd.DataFrame):
        if '날짜' not in data.columns or '내용' not in data.columns:
            logger.warning(f"[{data_type}] 데이터프레임에 필수 컬럼(날짜, 내용)이 없습니다.")
            return
        for _, row in data.iterrows():
            standardized_data.append({
                'date': str(row['날짜'])[:10], # YYYY-MM-DD
                'text': str(row['내용'])
            })
    elif data_type == "script" and isinstance(data, list):
        standardized_data = data
    
    if not standardized_data:
        logger.warning(f"[{data_type}] 트렌드 분석을 위한 데이터가 없습니다.")
        return

    try:
        okt = Okt()
        processed_docs = [] # {'month': 'YYYY-MM', 'text': 'word1 word2 ...'}

        for idx, item in enumerate(standardized_data):
            text = item.get('text', '')
            date_str = item.get('date', '')
            
            if not text or not date_str:
                continue
                
            month = date_str[:7]
            
            # 형태소 분석
            tagged_words = okt.pos(text, stem=True)
            # 불용어 필터링
            tokens = [word for word, tag in tagged_words 
                      if tag in ['Noun', 'Adjective', 'Verb'] and word not in stopwords]
            
            if len(tokens) >= 2:
                processed_docs.append({
                    'month': month,
                    'text': " ".join(tokens)
                })

        if not processed_docs:
            logger.warning(f"[{data_type}] 분석 가능한 텍스트 데이터가 없습니다.")
            return

        df_proc = pd.DataFrame(processed_docs)
        
        # 전체 텍스트 리스트
        all_texts = df_proc['text'].tolist()

        # 공통 설정
        token_pattern = r"(?u)\b\w+\b"
        ngram_range = (2, 2)

        # ---------------------------------------------------------
        # (A) 빈도수(Frequency) 기준 분석
        # ---------------------------------------------------------
        logger.info(f"[{data_type}] 빈도수 기준 상위 10개 키워드 추출 중...")
        vec_freq = CountVectorizer(ngram_range=ngram_range, min_df=1, token_pattern=token_pattern)
        X_freq = vec_freq.fit_transform(all_texts)
        sum_freq = np.array(X_freq.sum(axis=0)).flatten()
        words_freq = vec_freq.get_feature_names_out()
        
        # 상위 10개 인덱스
        top_k = 10
        if len(words_freq) < top_k:
            top_k = len(words_freq)
            
        top_idx_freq = sum_freq.argsort()[::-1][:top_k]
        top_keywords_freq = words_freq[top_idx_freq]
        logger.info(f" -> 빈도수 Top {top_k}: {top_keywords_freq}")

        # ---------------------------------------------------------
        # (B) TF-IDF 기준 분석
        # ---------------------------------------------------------
        logger.info(f"[{data_type}] TF-IDF 기준 상위 10개 키워드 추출 중...")
        vec_tfidf = TfidfVectorizer(ngram_range=ngram_range, min_df=1, token_pattern=token_pattern)
        X_tfidf = vec_tfidf.fit_transform(all_texts)
        sum_tfidf = np.array(X_tfidf.sum(axis=0)).flatten()
        words_tfidf = vec_tfidf.get_feature_names_out()
        
        top_idx_tfidf = sum_tfidf.argsort()[::-1][:top_k]
        top_keywords_tfidf = words_tfidf[top_idx_tfidf]
        logger.info(f" -> TF-IDF Top {top_k}: {top_keywords_tfidf}")

        # ---------------------------------------------------------
        # (C) 월별 데이터 집계 및 그래프 생성 함수
        # ---------------------------------------------------------
        def create_trend_chart(keywords, label_type):
            if len(keywords) == 0:
                return

            # 월별로 해당 키워드들의 빈도수를 다시 계산
            trend_data = []
            grouped = df_proc.groupby('month')
            
            # 키워드 매핑 (word -> index)
            vocab = {word: i for i, word in enumerate(keywords)}
            
            for month, group in grouped:
                month_texts = group['text'].tolist()
                # 고정된 어휘집(vocab)을 사용하여 카운트
                cv_month = CountVectorizer(vocabulary=vocab, ngram_range=ngram_range, token_pattern=token_pattern)
                dtm_month = cv_month.fit_transform(month_texts)
                counts = dtm_month.sum(axis=0).A1
                
                row = {'Month': month}
                for word, count in zip(keywords, counts):
                    row[word] = count
                trend_data.append(row)
            
            if not trend_data:
                return

            trend_df = pd.DataFrame(trend_data).set_index('Month').sort_index()
            
            # 그래프 그리기
            font_path = get_font_path()
            font_prop = None
            if font_path:
                font_prop = fm.FontProperties(fname=font_path)
            
            plt.rcParams['axes.unicode_minus'] = False 

            plt.figure(figsize=(12, 6))
            for keyword in keywords:
                if keyword in trend_df.columns:
                    plt.plot(trend_df.index, trend_df[keyword], marker='o', label=keyword)

            title_text = f'[{data_type}] 월별 상위 키워드 변화 ({label_type} Top {top_k}){title_suffix}'
            
            if font_prop:
                plt.title(title_text, fontsize=15, fontproperties=font_prop)
                plt.xlabel('월 (Month)', fontproperties=font_prop)
                plt.ylabel('빈도수 (Frequency)', fontproperties=font_prop)
                plt.legend(prop=font_prop)
                plt.xticks(rotation=45, fontproperties=font_prop)
            else:
                plt.title(title_text, fontsize=15)
                plt.xlabel('월 (Month)')
                plt.ylabel('빈도수 (Frequency)')
                plt.legend()
                plt.xticks(rotation=45)

            plt.grid(True)
            plt.tight_layout()

            # 파일명 구분 (freq vs tfidf)
            safe_label = "freq" if "빈도수" in label_type else "tfidf"
            file_name = f"{output_base_name}_trend_{data_type}_{safe_label}.png"
            plt.savefig(file_name)
            logger.info(f"[{data_type}] {label_type} 트렌드 그래프 저장 완료: {file_name}")
            plt.close()

        # 그래프 생성 실행
        create_trend_chart(top_keywords_freq, "빈도수 기준")
        create_trend_chart(top_keywords_tfidf, "TF-IDF 기준")

    except Exception as e:
        logger.error(f"[{data_type}] 트렌드 분석 중 오류 발생: {e}")

def sanitize_filename(filename):
    """파일 이름으로 사용할 수 없는 문자를 제거합니다."""
    return re.sub(r'[\\/*?:"<>|]', "", filename)


# ==========================================
# 3. 메인 실행 함수 (캡슐화)
# ==========================================
def main(
    api_key, 
    search_keyword, 
    max_videos=5, 
    max_comments_per_video=300, 
    search_start_date=None, 
    search_end_date=None, 
    collect_script=True,
    output_filename=None, # (선택) 저장할 파일 이름
    output_folder=None,   # (선택) 저장할 폴더 경로
    stopwords_file=None   # (선택) 불용어 파일 경로
):
    """
    YouTube 댓글 및 스크립트 수집기 메인 함수
    """
    
    # API 키 확인
    if not api_key or api_key == "여기에_본인의_API_KEY를_입력하세요":
        logger.error("오류: API 키를 설정해주세요.")
        return

    # 불용어 로드
    stopwords = load_stopwords(stopwords_file)

    # (1) 파일 저장 경로 결정 로직을 상단으로 이동
    if output_filename:
        output_base = os.path.splitext(output_filename)[0]
        csv_filename = output_filename if output_filename.lower().endswith('.csv') else f"{output_filename}.csv"
    else:
        if isinstance(search_keyword, list):
            # 키워드가 리스트인 경우 + 로 연결
            safe_keyword = "+".join(search_keyword).replace(" ", "_")
        else:
            safe_keyword = search_keyword.replace(" ", "_")
            
        output_base = f"youtube_comments_{safe_keyword}"
        csv_filename = f"{output_base}.csv"

    extracted_folder_name = None
    if output_folder:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
            logger.info(f"폴더 생성: {output_folder}")
        
        extracted_folder_name = os.path.basename(os.path.normpath(output_folder))
        csv_filename = os.path.join(output_folder, csv_filename)
        output_base = os.path.join(output_folder, output_base)

    # (2) 기존 CSV 파일 존재 여부 확인
    df = None
    target_videos = []
    all_scripts = []

    if os.path.exists(csv_filename):
        logger.info(f"기존 데이터 파일 발견: {csv_filename}")
        logger.info("API 수집을 건너뛰고 기존 데이터를 사용하여 분석합니다.")
        
        try:
            df = pd.read_csv(csv_filename)
            df['날짜'] = pd.to_datetime(df['날짜'])
            
            # Title Analysis를 위한 target_videos 정보 재구성 (있는 데이터 한에서)
            # 필수: published_at, title
            if 'Video_ID' in df.columns and '영상제목' in df.columns and '영상게시일' in df.columns:
                unique_vids = df.drop_duplicates(subset=['Video_ID'])
                for _, row in unique_vids.iterrows():
                    target_videos.append({
                        'id': row['Video_ID'],
                        'title': row['영상제목'],
                        'published_at': str(row['영상게시일']), # datetime to string
                        'channel': row['채널명'] if '채널명' in df.columns else 'Unknown',
                        'view_count': row['영상조회수'] if '영상조회수' in df.columns else 0
                    })
                logger.info(f"로드된 데이터: {len(df)}개 댓글, {len(target_videos)}개 영상 정보 복원")
            else:
                logger.warning("CSV 파일에 필수 컬럼(Video_ID, 영상제목, 영상게시일)이 없어 영상 정보를 복원할 수 없습니다.")
        
        except Exception as e:
            logger.error(f"기존 파일 로드 중 오류 발생: {e}")
            return
            
    else:
        # (3) 파일이 없으면 API 수집 진행 (기존 로직)
        
        # YouTube API 클라이언트 빌드
        youtube = build("youtube", "v3", developerKey=api_key)

        # 리스트 형태의 키워드 처리 및 로깅
        if isinstance(search_keyword, list):
            display_keyword = ", ".join(search_keyword)
        else:
            display_keyword = search_keyword
            
        logger.info(f"['{display_keyword}'] 키워드로 영상을 검색합니다...")
        
        # 1. 영상 검색
        target_videos = search_videos(youtube, search_keyword, max_videos, search_start_date, search_end_date)
        
        if not target_videos:
            logger.warning("검색된 영상이 없습니다.")
            return

        logger.info(f"총 {len(target_videos)}개의 영상을 찾았습니다. 수집을 시작합니다.")
        
        for idx, v in enumerate(target_videos, 1):
            logger.info(f"{idx}. [{v['channel']}] {v['title'][:40]}... (조회수: {v['view_count']})")
        
        all_comments = []
        all_scripts = [] 

        # 2. 각 영상별 댓글 및 스크립트 수집
        for video_info in target_videos:
            video_id = video_info['id']
            title = video_info['title']
            
            logger.info(f"\n[영상] {title[:40]}...")
            
            # (1) 댓글 수집
            video_comments = get_comments(youtube, video_info, max_comments_per_video)
            if video_comments:
                all_comments.extend(video_comments)
                logger.info(f"    -> 댓글 수집 완료 ({len(video_comments)}개)")

            # (2) 스크립트 수집
            if collect_script:
                logger.info(f"    -> 스크립트(자막) 확인 중...")
                script_text = get_video_script(video_id)
                
                if script_text:
                    script_data = {
                        'text': script_text,
                        'date': video_info['published_at'] # 영상 게시일 사용
                    }
                    all_scripts.append(script_data)

                    safe_title = sanitize_filename(title[:20])
                    if output_folder:
                        if not os.path.exists(output_folder):
                            os.makedirs(output_folder, exist_ok=True)
                        filename = os.path.join(output_folder, f"script_{safe_title}_{video_id}.txt")
                    else:
                        filename = f"script_{safe_title}_{video_id}.txt"
                    
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(f"Title: {title}\n")
                        f.write(f"Channel: {video_info['channel']}\n")
                        f.write(f"Views: {video_info['view_count']}\n")
                        f.write(f"Link: https://www.youtube.com/watch?v={video_id}\n")
                        f.write("="*30 + "\n\n")
                        f.write(script_text)
                    logger.info(f"    -> 스크립트 저장 완료: {filename}")
                else:
                    logger.warning("    -> 자막이 없거나 가져올 수 없습니다.")

            time.sleep(0.5) 
            
        # 댓글 데이터프레임 생성 및 저장
        if all_comments:
            df = pd.DataFrame(all_comments)
            df['날짜'] = pd.to_datetime(df['날짜'])
            
            # 줄바꿈(\n, \r), 탭(\t) 문자를 공백으로 대체
            df['내용'] = df['내용'].astype(str).str.replace(r'[\n\r\t]', ' ', regex=True)

            cols = ['영상제목', '채널명', '영상조회수', '영상게시일', '작성자', '내용', '날짜', '좋아요수', 'Video_ID']
            df = df.reindex(columns=cols)
            
            # [수정] 콤마 등이 포함된 데이터 보호를 위해 quoting 옵션 추가, escapechar 추가
            df.to_csv(csv_filename, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL, escapechar='\\')
            logger.info(f"댓글 수집 완료! 총 {len(df)}개의 댓글이 '{csv_filename}' 파일로 저장되었습니다.")
        else:
            logger.warning("수집된 댓글이 없습니다.")

    # (4) 공통 분석 단계
    # (1) 댓글 분석
    if df is not None and not df.empty:
        # 통계 분석
        generate_statistics(df, len(target_videos), output_base)

        # 댓글 워드 클라우드 (TF-IDF & Frequency 둘 다 생성)
        generate_wordcloud(df['내용'], output_base, data_type="comment", stopwords=stopwords, folder_name=extracted_folder_name)

        # 댓글 월별 키워드 트렌드 (2-gram)
        generate_monthly_trend(df, output_base, data_type="comment", stopwords=stopwords, folder_name=extracted_folder_name)

        if 'display' in globals():
            print("\n[댓글 데이터 미리보기]")
            display(df.head())
        else:
            print(df.head())
    
    # (2) 스크립트 분석 (기존 데이터 로드 시에는 all_scripts가 비어있으므로 스킵됨)
    if all_scripts:
        logger.info("\n수집된 스크립트를 기반으로 분석을 시작합니다.")
        # 스크립트 워드 클라우드 (TF-IDF & Frequency)
        generate_wordcloud(all_scripts, output_base, data_type="script", stopwords=stopwords, folder_name=extracted_folder_name)
        
        # 스크립트 월별 키워드 트렌드 (2-gram)
        generate_monthly_trend(all_scripts, output_base, data_type="script", stopwords=stopwords, folder_name=extracted_folder_name)
    elif not os.path.exists(csv_filename): # 수집을 시도했으나 없었던 경우에만 로그 출력
        logger.warning("수집된 스크립트가 없어 스크립트 분석을 수행할 수 없습니다.")

    # (3) 영상 제목 분석 (target_videos가 복원되었거나 수집된 경우 실행)
    if target_videos:
        logger.info("\n수집된 영상 제목을 기반으로 분석을 시작합니다.")
        titles_data = []
        for v in target_videos:
            titles_data.append({
                '날짜': v['published_at'],
                '내용': v['title']
            })
        
        title_df = pd.DataFrame(titles_data)
        
        # 제목 워드 클라우드
        generate_wordcloud(title_df['내용'], output_base, data_type="title", stopwords=stopwords, folder_name=extracted_folder_name)
        
        # 제목 월별 키워드 트렌드
        generate_monthly_trend(title_df, output_base, data_type="title", stopwords=stopwords, folder_name=extracted_folder_name)


# ==========================================
# 4. 실행 예시 (직접 실행 시)
# ==========================================
if __name__ == "__main__":
    # 사용자 설정
    MY_API_KEY = "여기에_본인의_API_KEY를_입력하세요"
    KEYWORD = ["아이폰 15", "갤럭시 S24"] # OR 검색 예시
    STOPWORDS_FILE = "stopwords.txt" # 불용어 파일 경로 예시

    # 함수 실행
    main(
        api_key=MY_API_KEY,
        search_keyword=KEYWORD,
        max_videos=5,
        max_comments_per_video=300,
        search_start_date=None, # 예: "2023-01-01"
        search_end_date=None,
        collect_script=True,
        output_filename=None, # 예: "my_result.csv"
        output_folder="./result", # 예: "./result" (저장할 폴더 지정)
        stopwords_file=STOPWORDS_FILE # 불용어 파일 경로 전달
    )