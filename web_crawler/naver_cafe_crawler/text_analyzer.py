import re
import logging
from collections import Counter

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_STOPWORDS = {
    '이', '가', '을', '를', '은', '는', '의', '에', '에서', '으로', '로',
    '와', '과', '도', '만', '도', '라', '이라', '하다', '있다', '없다',
    '되다', '하고', '해서', '그', '저', '이것', '저것', '그것', '것',
    '수', '때', '더', '좀', '잘', '못', '안', '다', '고', '나', '너',
    '우리', '제', '저는', '저도', '저도', '저희', '그냥', '근데', '그래도',
    '진짜', '정말', '너무', '많이', '같이', '같아요', '같은', '같아', '같습니다',
    '있어요', '있는', '없어요', '없는', '이제', '지금', '그게', '아', '음',
    '그리고', '그런데', '하지만', '근데', '그래서', '또', '다시', '어떻게',
}


class TextAnalyzer:
    """네이버 카페 게시글/댓글 단어 빈도 분석기."""

    def __init__(self, stopwords: set | None = None, min_word_len: int = 2):
        """
        Args:
            stopwords:     추가 불용어 집합. None이면 기본 불용어만 사용.
            min_word_len:  포함할 최소 단어 길이.
        """
        self.stopwords = DEFAULT_STOPWORDS | (stopwords or set())
        self.min_word_len = min_word_len
        self._tokenize = self._build_tokenizer()

    # ──────────────────────────────────────────
    # 토크나이저
    # ──────────────────────────────────────────

    def _build_tokenizer(self):
        """konlpy.Okt 사용, 없으면 정규식 기반 단순 토크나이저로 fallback."""
        try:
            from konlpy.tag import Okt
            okt = Okt()
            logger.info("토크나이저: konlpy.Okt")

            def tokenize(text: str) -> list[str]:
                return [
                    word for word, pos in okt.pos(text, stem=True)
                    if pos in ('Noun', 'Verb', 'Adjective')
                    and len(word) >= self.min_word_len
                    and word not in self.stopwords
                ]
        except ImportError:
            logger.warning("konlpy 없음 — 정규식 기반 토크나이저 사용. 'pip install konlpy' 권장.")

            def tokenize(text: str) -> list[str]:
                words = re.findall(r'[가-힣]{2,}', text)
                return [w for w in words if w not in self.stopwords and len(w) >= self.min_word_len]

        return tokenize

    # ──────────────────────────────────────────
    # 내부 공통 분석
    # ──────────────────────────────────────────

    def _count_words(self, texts: list[str], keywords: list[str] | None = None) -> Counter:
        """keywords가 주어지면 그 중 하나라도 포함된 문장만 분석합니다."""
        counter: Counter = Counter()
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue
            if keywords:
                text_lower = text.lower()
                sentences = [s for s in re.split(r'[.!?\n]', text) if any(kw.lower() in text_lower for kw in keywords)]
                if not sentences:
                    continue
                text = ' '.join(sentences)
            tokens = self._tokenize(text)
            bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
            counter.update(bigrams)
        return counter

    def _to_df(self, counter: Counter) -> pd.DataFrame:
        df = pd.DataFrame(counter.most_common(), columns=['word', 'count'])
        df['rank'] = df['count'].rank(method='min', ascending=False).astype(int)
        return df[['rank', 'word', 'count']].reset_index(drop=True)

    # ──────────────────────────────────────────
    # 게시글 분석
    # ──────────────────────────────────────────

    def analyze_articles(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        keywords: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """게시글 DataFrame의 바이그램 빈도를 분석합니다.

        Args:
            df:       get_article_contents() 결과 DataFrame.
            columns:  분석할 열 이름 목록. 기본값은 ['title', 'content'].
            keywords: 문자열 또는 목록. 지정하면 키워드 중 하나라도 포함된 문장만 분석합니다.

        Returns:
            rank, word, count 열을 가진 DataFrame.
        """
        cols = columns or ['title', 'content']
        available = [c for c in cols if c in df.columns]
        if not available:
            raise ValueError(f"DataFrame에 분석할 열이 없습니다. 요청: {cols}, 실제: {list(df.columns)}")

        texts = []
        for col in available:
            for t in df[col].dropna():
                texts.append(t[:t.index('🏆')] if '🏆' in t else t)

        kw_list = [keywords] if isinstance(keywords, str) else keywords
        logger.info("게시글 분석 — 열: %s, 텍스트 수: %d, 키워드 필터: %s", available, len(texts), kw_list or '없음')
        counter = self._count_words(texts, keywords=kw_list)
        return self._to_df(counter)

    # ──────────────────────────────────────────
    # 댓글 분석
    # ──────────────────────────────────────────

    def analyze_comments(
        self,
        df: pd.DataFrame,
        column: str = 'content',
        keywords: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """댓글 DataFrame의 바이그램 빈도를 분석합니다.

        Args:
            df:       get_all_comments() 결과 DataFrame.
            column:   분석할 열 이름. 기본값은 'content'.
            keywords: 문자열 또는 목록. 지정하면 키워드 중 하나라도 포함된 문장만 분석합니다.

        Returns:
            rank, word, count 열을 가진 DataFrame.
        """
        if column not in df.columns:
            raise ValueError(f"'{column}' 열이 DataFrame에 없습니다. 실제 열: {list(df.columns)}")

        texts = df[column].dropna().tolist()
        kw_list = [keywords] if isinstance(keywords, str) else keywords
        logger.info("댓글 분석 — 텍스트 수: %d, 키워드 필터: %s", len(texts), kw_list or '없음')
        counter = self._count_words(texts, keywords=kw_list)
        return self._to_df(counter)

    # ──────────────────────────────────────────
    # 시각화
    # ──────────────────────────────────────────

    def plot_bar(
        self,
        freq_df: pd.DataFrame,
        top_n: int = 20,
        title: str = '단어 빈도',
        figsize: tuple = (12, 5),
        save_pdf: str | None = None,
    ):
        """상위 N개 단어를 막대그래프로 시각화합니다.

        Args:
            freq_df:  analyze_articles() 또는 analyze_comments() 반환값.
            top_n:    표시할 단어 수.
            save_pdf: 저장할 PDF 파일 경로. None이면 저장하지 않음.
        """
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        # 한글 폰트 설정
        korean_fonts = [f.name for f in fm.fontManager.ttflist if any(
            k in f.name for k in ('Malgun', 'NanumGothic', 'Apple SD', 'Noto Sans CJK')
        )]
        if korean_fonts:
            plt.rcParams['font.family'] = korean_fonts[0]
        plt.rcParams['axes.unicode_minus'] = False

        top = freq_df.head(top_n)
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(top['word'], top['count'])
        ax.set_title(title)
        ax.set_xlabel('단어')
        ax.set_ylabel('빈도')
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if save_pdf:
            from matplotlib.backends.backend_pdf import PdfPages
            with PdfPages(save_pdf) as pdf:
                pdf.savefig(fig)
            logger.info("PDF 저장: %s", save_pdf)
        plt.show()
        return fig

    def plot_wordcloud(
        self,
        freq_df: pd.DataFrame,
        top_n: int = 100,
        title: str = '',
        font_path: str | None = None,
        figsize: tuple = (10, 6),
    ):
        """워드클라우드를 시각화합니다.

        Args:
            freq_df:   analyze_articles() 또는 analyze_comments() 반환값.
            font_path: 한글 폰트 경로. None이면 시스템 폰트 자동 탐색.
        """
        try:
            from wordcloud import WordCloud
        except ImportError:
            raise ImportError("wordcloud가 설치되지 않았습니다. 'pip install wordcloud'를 실행하세요.")
        import matplotlib.pyplot as plt

        if font_path is None:
            import matplotlib.font_manager as fm
            candidates = [
                f.fname for f in fm.fontManager.ttflist
                if any(k in f.name for k in ('Malgun', 'NanumGothic', 'Apple SD', 'Noto Sans CJK'))
            ]
            font_path = candidates[0] if candidates else None

        freq_dict = dict(zip(freq_df.head(top_n)['word'], freq_df.head(top_n)['count']))
        wc = WordCloud(
            font_path=font_path,
            background_color='white',
            width=800, height=500,
        ).generate_from_frequencies(freq_dict)

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(wc.to_image())
        ax.axis('off')
        if title:
            ax.set_title(title)
        plt.tight_layout()
        plt.show()
        return fig
