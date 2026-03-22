import json
import logging
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class NaverCafeCrawler:
    """네이버 카페 게시판 크롤러"""

    BASE_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
    }

    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(self.BASE_HEADERS)
        self.club_id: str | None = None
        self.cafe_name: str = ''
        self.cafe_slug: str = ''
        self._playwright = None
        self._browser = None
        self._context = None

    # ──────────────────────────────────────────
    # 인증
    # ──────────────────────────────────────────

    def login(self, naver_id: str, naver_pw: str) -> bool:
        """
        ID/PW로 네이버에 로그인합니다.
        ⚠️ 2단계 인증 계정은 set_cookies()를 사용하세요.
        """
        login_page_url = "https://nid.naver.com/nidlogin.login"
        self.session.get(login_page_url)

        payload = {
            'svctype': '0', 'enctp': '1',
            'encpw': naver_pw, 'encnm': naver_id,
            'id': naver_id, 'pw': naver_pw,
            'postDataKey': '', 'url': 'https://www.naver.com',
            'smart_LEVEL': '-1', 'logintp': 'P_ID',
        }
        self.session.post(
            login_page_url, data=payload,
            headers={'Referer': login_page_url, 'Content-Type': 'application/x-www-form-urlencoded'},
            allow_redirects=True,
        )

        cookies = self.session.cookies.get_dict()
        if 'NID_AUT' in cookies and 'NID_SES' in cookies:
            logger.info("로그인 성공")
            return True
        logger.warning("로그인 실패 - set_cookies()를 사용하세요")
        return False

    def set_cookies(self, nid_aut: str, nid_ses: str) -> None:
        """
        브라우저에서 복사한 쿠키로 세션을 설정합니다.
        개발자도구 → Application → Cookies → naver.com 에서 NID_AUT, NID_SES를 복사하세요.
        """
        self.session.cookies.set('NID_AUT', nid_aut, domain='.naver.com')
        self.session.cookies.set('NID_SES', nid_ses, domain='.naver.com')
        logger.info("쿠키 세션 설정 완료")

    # ──────────────────────────────────────────
    # 카페 정보
    # ──────────────────────────────────────────

    def load_cafe(self, cafe_url: str) -> None:
        """카페 URL에서 club_id와 카페명을 추출해 인스턴스에 저장합니다."""
        res = self.session.get(cafe_url)
        soup = BeautifulSoup(res.text, 'html.parser')

        club_id = None

        if 'clubid' in res.url.lower():
            m = re.search(r'clubid=([\d]+)', res.url, re.IGNORECASE)
            if m:
                club_id = m.group(1)

        if not club_id:
            m = re.search(r'"clubId"\s*:\s*"?(\d+)"?', res.text)
            if m:
                club_id = m.group(1)

        if not club_id:
            # 페이지 내 메뉴 링크의 search.clubid= 에서 추출 (신규 SPA 구조 대응)
            m = re.search(r'search\.clubid=(\d+)', res.text, re.IGNORECASE)
            if m:
                club_id = m.group(1)

        if not club_id:
            m = re.search(r'cafe\.naver\.com/([^/?]+)', cafe_url)
            if m:
                api_res = self.session.get(f"https://cafe.naver.com/CafeGateway.nhn?cafeUrl={m.group(1)}")
                m2 = re.search(r'"clubId"\s*:\s*(\d+)', api_res.text)
                if m2:
                    club_id = m2.group(1)

        name_tag = soup.select_one('h1.d-café-name, .cafe-name, #cafe-name, title')
        self.cafe_name = name_tag.get_text(strip=True) if name_tag else cafe_url.rstrip('/').split('/')[-1]
        self.club_id = club_id
        m_slug = re.search(r'cafe\.naver\.com/([^/?#]+)', cafe_url)
        self.cafe_slug = m_slug.group(1) if m_slug else ''

        logger.info("카페명: %s", self.cafe_name)
        logger.info("Club ID: %s", self.club_id)
        logger.info("Cafe Slug: %s", self.cafe_slug)

    # ──────────────────────────────────────────
    # 메뉴 목록
    # ──────────────────────────────────────────

    def find_menu_id(self, name: str, menus: list) -> str | None:
        """메뉴 이름(부분 일치)으로 menu_id를 반환합니다."""
        matched = [m for m in menus if name in m['name']]
        if not matched:
            logger.warning("'%s'과 일치하는 메뉴가 없습니다.", name)
            return None
        if len(matched) > 1:
            logger.warning("'%s'과 일치하는 메뉴가 여러 개입니다:", name)
            for m in matched:
                logger.warning("  [%s] %s", m['menu_id'], m['name'])
            return None
        return matched[0]['menu_id']

    def get_menus(self) -> list:
        """카페 메뉴(게시판) 목록을 반환합니다. menu_id와 이름을 확인할 수 있습니다."""
        if not self.cafe_slug:
            logger.error("cafe_slug가 없습니다. load_cafe()를 먼저 호출하세요.")
            return []

        res = self.session.get(
            f"https://cafe.naver.com/{self.cafe_slug}",
            headers={'Referer': 'https://cafe.naver.com/'},
        )

        # __NEXT_DATA__ JSON에서 메뉴 추출
        m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', res.text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
                menus = self._extract_menus_from_next_data(data)
                if menus:
                    for menu in menus:
                        logger.info("  [%s] %s (%s)", menu['menu_id'], menu['name'], menu['menu_type'])
                    return menus
            except Exception:
                pass

        # fallback: BeautifulSoup으로 search.menuid가 포함된 모든 <a> 태그 파싱
        soup = BeautifulSoup(res.text, 'html.parser')
        seen, menus = set(), []
        for a in soup.find_all('a', href=re.compile(r'search\.menuid=\d+')):
            href = a.get('href', '')
            m = re.search(r'search\.menuid=(\d+)', href)
            bt = re.search(r'search\.boardtype=(\w+)', href)
            if not m:
                continue
            mid = m.group(1)
            if mid in seen:
                continue
            seen.add(mid)
            name = a.get('title') or a.get_text(strip=True)
            menus.append({
                'menu_id':   mid,
                'name':      name,
                'menu_type': bt.group(1) if bt else '',
            })

        if not menus:
            logger.warning("메뉴 목록을 자동으로 가져올 수 없습니다.")
            logger.warning("브라우저에서 게시판 클릭 후 URL의 search.menuid= 값을 직접 확인하세요.")
            return []

        return menus

    def _extract_menus_from_next_data(self, data: dict) -> list:
        """__NEXT_DATA__ JSON을 재귀 탐색하여 메뉴 목록을 추출합니다."""
        menus = []

        def search(obj):
            if isinstance(obj, dict):
                # menuId 또는 menuNo 키가 있으면 메뉴 항목으로 간주
                if ('menuId' in obj or 'menuNo' in obj) and ('menuName' in obj or 'name' in obj):
                    menus.append({
                        'menu_id':   str(obj.get('menuId') or obj.get('menuNo', '')),
                        'name':      obj.get('menuName') or obj.get('name', ''),
                        'menu_type': obj.get('menuType') or obj.get('type', ''),
                    })
                for v in obj.values():
                    search(v)
            elif isinstance(obj, list):
                for item in obj:
                    search(item)

        search(data)
        # 중복 제거
        seen, result = set(), []
        for m in menus:
            if m['menu_id'] not in seen:
                seen.add(m['menu_id'])
                result.append(m)
        return result

    # ──────────────────────────────────────────
    # 게시글 목록
    # ──────────────────────────────────────────

    def _fetch_article_list_page(self, menu_id: int, page: int, per_page: int = 50) -> tuple[list, bool]:
        """(articles, has_next) 튜플 반환"""
        res = self.session.get(
            'https://apis.naver.com/cafe-web/cafe2/ArticleListV2.json',
            params={
                'search.clubid':  self.club_id,
                'search.menuid':  menu_id,
                'search.page':    page,
                'userDisplay':    per_page,
            },
            headers={'Referer': f'https://cafe.naver.com/{self.cafe_slug}'},
        )
        try:
            result = res.json().get('message', {}).get('result', {})
        except Exception:
            return [], False

        articles = []
        for a in result.get('articleList', []):
            article_id = str(a.get('articleId', ''))
            writer = a.get('writerInfo') or {}
            ts = a.get('writeDateTimestamp', '') or a.get('writeDate', '')
            try:
                date_str = datetime.fromtimestamp(int(ts) / 1000).strftime('%Y-%m-%d')
            except (TypeError, ValueError):
                date_str = str(ts)
            articles.append({
                'article_id':    article_id,
                'title':         a.get('subject', ''),
                'author':        writer.get('nick', '') if isinstance(writer, dict) else str(writer),
                'date':          date_str,
                'view_count':    str(a.get('readCount', '')),
                'comment_count': str(a.get('commentCount', '0')),
                'url':           f"https://cafe.naver.com/{self.cafe_slug}/{article_id}",
            })
        return articles, result.get('hasNext', False)

    def _parse_article_rows(self, rows) -> list:
        articles = []
        for row in rows:
            try:
                if 'notice' in row.get('class', []):
                    continue
                title_tag = row.select_one('a.article, td.td_article a, .article-title a')
                if not title_tag:
                    continue
                href = title_tag.get('href', '')
                m = re.search(r'articleid=(\d+)', href)
                author_tag = row.select_one('.td_name a, .nick, .writer')
                date_tag   = row.select_one('.td_date, .date, time')
                view_tag   = row.select_one('.td_view, .view')
                comment_tag = row.select_one('.td_comment, .comment_count')
                articles.append({
                    'article_id':    m.group(1) if m else '',
                    'title':         title_tag.get_text(strip=True),
                    'author':        author_tag.get_text(strip=True) if author_tag else '',
                    'date':          date_tag.get_text(strip=True) if date_tag else '',
                    'view_count':    view_tag.get_text(strip=True) if view_tag else '',
                    'comment_count': comment_tag.get_text(strip=True) if comment_tag else '0',
                    'url':           f"https://cafe.naver.com{href}" if href.startswith('/') else href,
                })
            except Exception:
                continue
        return articles

    def get_article_count(self, menu_id: int = 0) -> int:
        """게시판 전체 게시글 수를 반환합니다. (perPage=1 이진 탐색 — 마지막 페이지 번호 = 총 게시글 수)"""
        lo, hi = 1, 1
        # 상한선 탐색
        while True:
            _, has_next = self._fetch_article_list_page(menu_id, hi, per_page=1)
            if not has_next:
                break
            hi *= 2
            time.sleep(0.3)

        # 이진 탐색으로 게시글이 있는 마지막 페이지 탐색
        while lo < hi:
            mid = (lo + hi + 1) // 2
            articles, _ = self._fetch_article_list_page(menu_id, mid, per_page=1)
            if articles:
                lo = mid
            else:
                hi = mid - 1
            time.sleep(0.3)

        logger.info("총 게시글 수: %d개", lo)
        return lo

    def get_articles(
        self,
        menu_id: int = 0,
        max_pages: int = 3,
        start_date: str | None = None,
        end_date: str | None = None,
        menu_name: str = '',
    ) -> list:
        """게시판 글 목록을 수집합니다.

        Args:
            start_date: 수집 시작일 (포함). 형식: 'YYYY-MM-DD'
            end_date:   수집 종료일 (포함). 형식: 'YYYY-MM-DD'
            menu_name:  게시판 이름. 지정하면 각 게시글에 menu_name 열이 추가됩니다.
        """
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000) if start_date else None
        end_ts   = int(datetime.strptime(end_date,   '%Y-%m-%d').timestamp() * 1000) + 86400_000 - 1 if end_date else None

        all_articles = []
        for page in range(1, max_pages + 1):
            logger.info("페이지 %d/%d 수집 중...", page, max_pages)
            articles, has_next = self._fetch_article_list_page(menu_id, page)
            if not articles:
                logger.info("더 이상 게시글이 없습니다.")
                break

            stop = False
            for a in articles:
                date_val = a.get('date', '')
                try:
                    ts = int(datetime.strptime(date_val, '%Y-%m-%d').timestamp() * 1000)
                except (TypeError, ValueError):
                    ts = None

                if start_ts and ts is not None and ts < start_ts:
                    stop = True  # 날짜순 정렬이므로 이후는 모두 범위 밖
                    break
                if end_ts and ts is not None and ts > end_ts:
                    continue
                if menu_name:
                    a['menu_name'] = menu_name
                all_articles.append(a)

            logger.info("페이지 %d: %d개 처리 (누적: %d개)", page, len(articles), len(all_articles))

            if stop:
                logger.info("start_date 이전 게시글 도달, 수집 종료.")
                break
            if not has_next:
                logger.info("마지막 페이지입니다.")
                break
            time.sleep(self.delay)
        return all_articles

    # ──────────────────────────────────────────
    # Playwright 브라우저
    # ──────────────────────────────────────────

    def _run_in_thread(self, fn, *args, **kwargs):
        """sync_playwright는 asyncio 루프(Jupyter 등)에서 실행 불가 — 별도 스레드에서 실행합니다."""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(fn, *args, **kwargs).result()

    def _fetch_content_batch(self, targets: list) -> list:
        """Playwright로 게시글 본문을 일괄 수집합니다. 스레드 내부에서 실행됩니다."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise RuntimeError("playwright가 설치되지 않았습니다. 'pip install playwright && playwright install chromium'을 실행하세요.")

        CONTENT_SELECTORS = [
            '.se-main-container', '.ArticleContentsArea', '#tbody', '.article_body', '.content_area',
        ]
        cookies = [
            {'name': c.name, 'value': c.value, 'domain': c.domain or '.naver.com', 'path': '/'}
            for c in self.session.cookies
        ]

        contents = []
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            ctx = browser.new_context(user_agent=self.BASE_HEADERS['User-Agent'], locale='ko-KR')
            if cookies:
                ctx.add_cookies(cookies)
            page = ctx.new_page()

            for i, article in enumerate(targets):
                article_id = article['article_id']
                result = {
                    'article_id': article_id,
                    'title': '', 'author': '', 'date': '',
                    'content': '', 'view_count': '', 'like_count': '',
                }
                try:
                    logger.info("[%d/%d] 본문 수집: %s...", i + 1, len(targets), article.get('title', '')[:30])
                    page.goto(
                        f"https://cafe.naver.com/{self.cafe_slug}/{article_id}",
                        wait_until='networkidle',
                        timeout=30_000,
                    )

                    # 본문은 ca-fe/cafes/{club_id}/articles/{article_id} iframe 안에 있음
                    article_frame = None
                    for frame in page.frames:
                        if f'/articles/{article_id}' in frame.url:
                            article_frame = frame
                            break

                    if article_frame is None:
                        logger.warning("본문 iframe 없음 (article_id=%s)", article_id)
                        contents.append({**article, **result})
                        time.sleep(self.delay)
                        continue

                    # iframe 안에서 본문 컨테이너 대기
                    for sel in CONTENT_SELECTORS:
                        try:
                            article_frame.wait_for_selector(sel, timeout=8_000)
                            break
                        except Exception:
                            continue

                    def _first_text(frame, selectors):
                        for sel in selectors:
                            el = frame.query_selector(sel)
                            if el:
                                return el.inner_text().strip()
                        return ''

                    result['title']      = _first_text(article_frame, ['h3.title', '.ArticleTitle', '.tit_h3', 'h2.title'])
                    result['author']     = _first_text(article_frame, ['.WriterInfo .nickname', '.article_writer .nick', '.writer_nick', '.member_nick', '.nickname'])
                    result['date']       = _first_text(article_frame, ['.article_info .date', '.WriterInfo .date', 'time', '.date'])
                    result['view_count'] = re.sub(r'[^\d]', '', _first_text(article_frame, ['.article_info .view', '.count_view', '.view_count']))
                    result['like_count'] = re.sub(r'[^\d]', '', _first_text(article_frame, ['.u_likeit_text', '.like_count', '.cnt_like']))

                    for sel in CONTENT_SELECTORS:
                        el = article_frame.query_selector(sel)
                        if el:
                            result['content'] = el.inner_text().strip()
                            break

                    if not result['content']:
                        logger.warning("본문 파싱 실패 (article_id=%s)", article_id)
                except Exception as e:
                    logger.error("본문 수집 오류 (article_id=%s): %s", article_id, e)

                contents.append({**article, **result})
                time.sleep(self.delay)

            page.close()
            browser.close()
        return contents

    def close(self) -> None:
        """하위 호환성을 위해 남겨둔 메서드입니다. 현재는 별도 정리가 필요하지 않습니다."""
        pass

    # ──────────────────────────────────────────
    # 게시글 본문
    # ──────────────────────────────────────────

    def debug_article(self, article_id: str) -> None:
        """게시글 수집 실패 원인을 진단합니다. 노트북에서 실행 후 결과를 확인하세요."""
        logger.debug("=" * 60)
        logger.debug("[1] Web API 응답")
        api_url = f"https://cafe.naver.com/ca-fe/web-api/v1/cafes/{self.club_id}/articles/{article_id}"
        res = self.session.get(api_url, headers={
            'Referer': f"https://cafe.naver.com/{self.cafe_slug}/{article_id}",
            'Accept': 'application/json',
        })
        logger.debug("  status: %d", res.status_code)
        try:
            data = res.json()
            logger.debug("  top-level keys: %s", list(data.keys()))
            result = data.get('result') or {}
            if result:
                logger.debug("  result keys: %s", list(result.keys()) if isinstance(result, dict) else type(result))
            logger.debug("  raw (500자): %s", res.text[:500])
        except Exception as e:
            logger.debug("  JSON 파싱 실패: %s", e)
            logger.debug("  raw (500자): %s", res.text[:500])

        logger.debug("[2] ArticleRead.nhn HTML 응답")
        iframe_url = f"https://cafe.naver.com/ArticleRead.nhn?clubid={self.club_id}&articleid={article_id}&boardtype=L"
        res2 = self.session.get(iframe_url, headers={
            'Referer': f"https://cafe.naver.com/{self.cafe_slug}/{article_id}",
        })
        logger.debug("  status: %d", res2.status_code)
        logger.debug("  final URL: %s", res2.url)
        soup = BeautifulSoup(res2.text, 'html.parser')
        iframes = soup.select('iframe')
        logger.debug("  iframe 개수: %d", len(iframes))
        for f in iframes:
            logger.debug("    src=%s", f.get('src', '')[:100])
        for sel in ['#tbody', '.article_body', '.ArticleContentsArea', '.se-main-container', '.content_area', '.se-component']:
            tag = soup.select_one(sel)
            logger.debug("  %s: %s", sel, '있음 (' + tag.get_text(strip=True)[:50] + ')' if tag else '없음')
        logger.debug("  raw (500자): %s", res2.text[:500])
        logger.debug("=" * 60)

    def get_article_content(self, article_id: str) -> dict:
        """게시글 본문을 Playwright로 수집합니다."""
        article = {'article_id': article_id, 'title': '', 'author': '', 'date': '', 'content': '', 'view_count': '', 'like_count': ''}
        results = self._run_in_thread(self._fetch_content_batch, [article])
        return results[0] if results else article

    def get_article_contents(self, articles: list, limit: int | None = None) -> list:
        """게시글 목록의 본문을 일괄 수집합니다. title에서 메뉴 이름을 분리해 menu_name 열을 추가합니다."""
        targets = articles[:limit] if limit else articles
        contents = self._run_in_thread(self._fetch_content_batch, targets)
        for item in contents:
            if '\n' in item.get('title', ''):
                menu_name, title = item['title'].split('\n', 1)
                item['menu_name'] = menu_name.strip()
                item['title'] = title.strip()
            else:
                item.setdefault('menu_name', '')
            content = item.get('content', '')
            if '🏆' in content:
                item['content'] = content[:content.index('🏆')].rstrip()
        return contents

    # ──────────────────────────────────────────
    # 댓글
    # ──────────────────────────────────────────

    def get_comments(self, article_id: str) -> list:
        """게시글 댓글을 Playwright로 수집합니다."""
        return self._run_in_thread(self._fetch_comments_batch, [{'article_id': article_id}])

    def _fetch_comments_batch(self, targets: list) -> list:
        """Playwright로 댓글을 일괄 수집합니다. 스레드 내부에서 실행됩니다."""
        from playwright.sync_api import sync_playwright

        cookies = [
            {'name': c.name, 'value': c.value, 'domain': c.domain or '.naver.com', 'path': '/'}
            for c in self.session.cookies
        ]
        all_comments = []
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            ctx = browser.new_context(user_agent=self.BASE_HEADERS['User-Agent'], locale='ko-KR')
            if cookies:
                ctx.add_cookies(cookies)
            page = ctx.new_page()

            for i, article in enumerate(targets):
                article_id = article['article_id']
                try:
                    logger.info("[%d/%d] 댓글 수집: article_id=%s", i + 1, len(targets), article_id)
                    page.goto(
                        f"https://cafe.naver.com/{self.cafe_slug}/{article_id}",
                        wait_until='networkidle',
                        timeout=30_000,
                    )
                    article_frame = next(
                        (f for f in page.frames if f'/articles/{article_id}' in f.url), None
                    )
                    if not article_frame:
                        logger.warning("댓글 iframe 없음 (article_id=%s)", article_id)
                        continue

                    try:
                        article_frame.wait_for_selector('.CommentItem', timeout=8_000)
                    except Exception:
                        logger.info("댓글 없음 (article_id=%s)", article_id)
                        continue

                    for item in article_frame.query_selector_all('.CommentItem'):
                        comment_id = item.get_attribute('id') or ''
                        # 답글 여부: 들여쓰기 클래스 또는 부모 구조로 판단
                        is_reply = 'reply' in (item.get_attribute('class') or '').lower()

                        author = ''
                        for sel in ['.comment_nickname', '.nick']:
                            el = item.query_selector(sel)
                            if el:
                                author = el.inner_text().strip()
                                break

                        content = ''
                        el = item.query_selector('.text_comment')
                        if el:
                            content = el.inner_text().strip()

                        date = ''
                        for sel in ['.comment_date', 'time', '.date']:
                            el = item.query_selector(sel)
                            if el:
                                date = el.inner_text().strip()
                                break

                        like_count = ''
                        for sel in ['.u_likeit_text', '.like_count', '.cnt_like']:
                            el = item.query_selector(sel)
                            if el:
                                like_count = re.sub(r'[^\d]', '', el.inner_text())
                                break

                        all_comments.append({
                            'article_id': article_id,
                            'comment_id': comment_id,
                            'author':     author,
                            'content':    content,
                            'date':       date,
                            'like_count': like_count,
                            'is_reply':   is_reply,
                        })
                except Exception as e:
                    logger.error("댓글 수집 오류 (article_id=%s): %s", article_id, e)
                time.sleep(self.delay)

            page.close()
            browser.close()
        return all_comments

    def get_all_comments(self, articles: list, limit: int | None = None) -> list:
        """댓글이 있는 게시글의 댓글을 일괄 수집합니다."""
        targets = [a for a in (articles[:limit] if limit else articles)
                   if str(a.get('comment_count', '0')).strip() not in ('', '0')]
        logger.info("댓글이 있는 게시글: %d개", len(targets))
        return self._run_in_thread(self._fetch_comments_batch, targets)

    # ──────────────────────────────────────────
    # 키워드 검색
    # ──────────────────────────────────────────

    def search_articles(
        self,
        keywords: str | list[str],
        max_pages: int = 3,
        start_date: str | None = None,
        end_date: str | None = None,
        oldest_first: bool = False,
    ) -> list:
        """카페 내 키워드 검색으로 게시글을 수집합니다. 여러 키워드는 각각 검색 후 합칩니다 (OR).

        Args:
            keywords:      단일 문자열 또는 키워드 목록.
            start_date:    수집 시작일 (포함). 형식: 'YYYY-MM-DD'
            end_date:      수집 종료일 (포함). 형식: 'YYYY-MM-DD'
            oldest_first:  True면 마지막 페이지부터 역순으로 탐색 (오래된 게시글 우선).
        """
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000) if start_date else None
        end_ts   = int(datetime.strptime(end_date,   '%Y-%m-%d').timestamp() * 1000) + 86400_000 - 1 if end_date else None

        kw_list = [keywords] if isinstance(keywords, str) else keywords
        seen, all_articles = set(), []

        def _fetch_page(keyword, page):
            res = self.session.get(
                f'https://apis.naver.com/cafe-web/cafe-search-api/v1.0/cafes/{self.club_id}/search/articles',
                params={'query': keyword, 'perPage': 15, 'page': page, 'menuId': 0},
                headers={'Referer': f'https://cafe.naver.com/{self.cafe_slug}'},
            )
            try:
                return res.json()
            except Exception:
                return {}

        def _parse_ts(add_date):
            if isinstance(add_date, str) and 'T' in add_date:
                try:
                    return int(datetime.strptime(add_date[:19], '%Y-%m-%dT%H:%M:%S').timestamp() * 1000)
                except ValueError:
                    pass
            elif add_date:
                try:
                    return int(add_date)
                except (TypeError, ValueError):
                    pass
            return None

        for keyword in kw_list:
            logger.info("키워드 검색: '%s' (방향: %s)", keyword, "오래된순" if oldest_first else "최신순")

            if oldest_first:
                # 총 페이지 수 파악을 위해 1페이지 먼저 요청
                probe = _fetch_page(keyword, 1)
                last_page = probe.get('result', {}).get('pageInfo', {}).get('lastNavigationPageNumber', 1)
                page_seq = range(last_page, max(last_page - max_pages, 0), -1)
            else:
                page_seq = range(1, max_pages + 1)

            for page in page_seq:
                data         = _fetch_page(keyword, page)
                result       = data.get('result', {})
                article_list = result.get('articleList', [])
                page_info    = result.get('pageInfo', {})
                logger.info("'%s' 페이지 %d / article_list 수: %d", keyword, page, len(article_list))

                if oldest_first:
                    article_list = list(reversed(article_list))

                stop = False
                new = []
                for entry in article_list:
                    a = entry.get('item', entry)
                    article_id = str(a.get('articleId') or a.get('id', ''))
                    if article_id in seen:
                        continue
                    add_date = a.get('addDate') or a.get('writeDateTimestamp') or a.get('writeDate', '')
                    ts = _parse_ts(add_date)

                    if oldest_first:
                        if end_ts and ts is not None and ts > end_ts:
                            stop = True
                            break
                        if start_ts and ts is not None and ts < start_ts:
                            continue
                    else:
                        if start_ts and ts is not None and ts < start_ts:
                            stop = True
                            break
                        if end_ts and ts is not None and ts > end_ts:
                            continue

                    writer = a.get('writerInfo') or a.get('writer') or {}
                    date_str = add_date[:10] if isinstance(add_date, str) and len(add_date) >= 10 else ''
                    new.append({
                        'article_id':    article_id,
                        'title':         a.get('subject') or a.get('title', ''),
                        'author':        writer.get('nickname') or writer.get('nick', '') if isinstance(writer, dict) else str(writer),
                        'date':          date_str,
                        'view_count':    str(a.get('readCount') or a.get('viewCount', '')),
                        'comment_count': str(a.get('commentCount', '0')),
                        'url':           f"https://cafe.naver.com/{self.cafe_slug}/{article_id}",
                    })
                    seen.add(article_id)

                all_articles.extend(new)
                logger.info("'%s' 페이지 %d: %d개 (누적: %d개)", keyword, page, len(new), len(all_articles))

                if stop:
                    break
                if not oldest_first and not page_info.get('visibleNextButton', False):
                    break
                time.sleep(self.delay)

        return all_articles

    # ──────────────────────────────────────────
    # 저장
    # ──────────────────────────────────────────

    def save(self, data: list, label: str, cafe_slug: str = '') -> str:
        """데이터를 CSV로 저장하고 파일명을 반환합니다."""
        if not data:
            logger.warning("저장할 %s 데이터가 없습니다.", label)
            return ''
        slug = cafe_slug or self.cafe_name or 'cafe'
        filename = f"cafe_{slug}_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame(data).to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info("%s 저장: %s (%d행)", label, filename, len(data))
        return filename
