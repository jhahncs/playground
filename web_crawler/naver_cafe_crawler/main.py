import logging
from pathlib import Path
import pandas as pd
from naver_cafe_crawler import NaverCafeCrawler
from text_analyzer import TextAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

# ==========================================
# 설정
# ==========================================
NID_AUT = "/taqHW5KZgjt8F5nJhrCOqCCRTiUaWGW0R007tfSaHCmfXseiR6aSkN/6kCeW9X/"
NID_SES = "AAABnkJrFh8NLjoKYgNeSp9Y1SmRks75KraQPP5qx897QszjUD3SnJSV/6gb5B+K9OThy3Yqw7Jiv78b80SHebnPTnhoy6X0elvvI9NmQKXmkcB1vaFHFKvbq5qw4EnsV2+D1bQwKbYlaEiIyvrA/FRTeKz9Vk7NRUmAeHKcmBiseqRIdCF0qOB+oxeELnl+04y1NP+HyDMzT1SPUxfXWwEiPBIYdE1yXbLHsXbNTWSxb6CNv9YFkYKbzsecfB/oqk7qzIMh/ubZ9cfroqSPpkVsizzNVnMZkN0MOpV0J0Z5G7KxPo4U/KUWe1Eh26WwbdT+unbgVtfBtftiskrrcZLEFKY4K+9d37904xOd8lpaW5hkggjM/jw5cHJeBnVAkkj0L2NLx4pPyObiBMP2DLwU5Io31dYDxWuWyC0lKPjnT0PU5NYvYZWqYW7Gv0kJBUwHQdgbXspYlPws8Nw0HYsiLaaWGKoasAraEaFcXBYx68v6tbUWzrPXxmrlig6l5YD6P292x4LOWo7wrm0bn8R/RF/MpicL6De7D4fvcT8VOdW+"

CAFE_URL      = "https://cafe.naver.com/gugrade"
MAX_PAGES     = 1000
CONTENT_LIMIT = 10000
DELAY         = 1.0
START_DATE    = "2025-01-01"
END_DATE      = "2025-12-31"
KEYWORDS      = ["PSAT", "피셋"]

OUTPUT_DIR    = Path("output")

MENUS = [
    "공드림 자유수다",
    "국가직 9급 수다방",
    "국가직 7급 수다방",
    "지방직 9급 수다방",
    "지방직 7급 수다방"
]
# ==========================================


def process_menu(crawler, analyzer, menu_id, menu_name):
    OUTPUT_DIR.mkdir(exist_ok=True)
    prefix = OUTPUT_DIR / f"{menu_id}_{menu_name}"
    logging.info("===== 메뉴 시작: %s (id=%d) =====", menu_name, menu_id)

    # ── 게시글 목록 수집 ──
    articles = crawler.get_articles(
        menu_id=menu_id,
        max_pages=MAX_PAGES,
        start_date=START_DATE,
        end_date=END_DATE,
        menu_name=menu_name,
    )
    logging.info("게시글 목록 수집 완료: %d개", len(articles))
    if not articles:
        logging.info("게시글이 없어 건너뜁니다.")
        return

    # ── 게시글 본문 수집 ──
    contents = crawler.get_article_contents(articles, limit=CONTENT_LIMIT)
    logging.info("게시글 본문 수집 완료: %d개", len(contents))
    contents_path = f"{prefix}_contents.csv"
    pd.DataFrame(contents).to_csv(contents_path, index=False, encoding="utf-8-sig")

    # ── 댓글 수집 ──
    comments = crawler.get_all_comments(articles, limit=CONTENT_LIMIT)
    logging.info("댓글 수집 완료: %d개", len(comments))
    comments_path = f"{prefix}_comments.csv"
    pd.DataFrame(comments).to_csv(comments_path, index=False, encoding="utf-8-sig")

    # ── 단어 빈도 분석 ──
    contents_df = pd.read_csv(contents_path)
    article_freq = analyzer.analyze_articles(contents_df, keywords=KEYWORDS)
    analyzer.plot_bar(
        article_freq, top_n=20,
        title=f"[{menu_name}] 게시글 단어 빈도",
        save_pdf=str(f"{prefix}_article_freq.pdf"),
    )

    comments_df = pd.read_csv(comments_path)
    comment_freq = analyzer.analyze_comments(comments_df, keywords=KEYWORDS)
    analyzer.plot_bar(
        comment_freq, top_n=20,
        title=f"[{menu_name}] 댓글 단어 빈도",
        save_pdf=str(f"{prefix}_comment_freq.pdf"),
    )

    logging.info("저장 완료: %s, %s", contents_path, comments_path)


def main():
    crawler = NaverCafeCrawler(delay=DELAY)
    crawler.set_cookies(NID_AUT, NID_SES)
    crawler.load_cafe(CAFE_URL)

    analyzer = TextAnalyzer()
    menus = crawler.get_menus()

    for menu_name in MENUS:
        menu_id = crawler.find_menu_id(menu_name, menus)
        if menu_id is None:
            logging.warning("메뉴를 찾을 수 없습니다: '%s' — 건너뜁니다.", menu_name)
            continue
        process_menu(crawler, analyzer, int(menu_id), menu_name)

    logging.info("===== 전체 완료 =====")


if __name__ == "__main__":
    main()
