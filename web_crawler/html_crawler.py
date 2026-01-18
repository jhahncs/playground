import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse
import re

def download_pdfs_from_url(base_url, download_dir='downloaded_pdfs'):
    """
    주어진 URL에서 모든 PDF 파일을 찾아 지정된 디렉터리에 다운로드합니다.
    """
    print(f"🔗 웹페이지 접속 시도: {base_url}")
    
    # 1. HTML 콘텐츠 가져오기
    try:
        response = requests.get(base_url, timeout=10)
        response.raise_for_status() # HTTP 오류가 발생하면 예외 발생
    except requests.exceptions.RequestException as e:
        print(f"❌ 웹페이지 접속 실패: {e}")
        return

    # 2. BeautifulSoup으로 HTML 파싱
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # 다운로드 디렉터리 생성
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        print(f"📁 다운로드 디렉터리 생성: '{download_dir}'")

    pdf_count = 0
    
    # 3. 모든 <a> 태그(링크) 찾기
    for link in soup.find_all('a', href=True):
        href = link.get('href')
        
        # PDF 파일 링크 필터링 (대소문자 구분 없이 .pdf로 끝나는지 확인)
        if href.lower().endswith('.pdf'):
            # 4. 절대 URL로 변환
            pdf_url = urljoin(base_url, href)
            
            # 파일 이름 결정
            # URL의 마지막 경로를 파일 이름으로 사용
            path = urlparse(pdf_url).path
            filename = os.path.basename(path).split('?')[0] # 쿼리 파라미터 제거
            
            # 파일 이름 정리 (안전하지 않은 문자 제거)
            filename = re.sub(r'[^\w\-_\. ]', '_', filename)
            
            if not filename:
                filename = f"unnamed_pdf_{pdf_count}.pdf" # 파일 이름이 없는 경우 대비

            save_path = os.path.join(download_dir, filename)

            # 이미 다운로드된 파일인지 확인 (선택 사항)
            if os.path.exists(save_path):
                print(f"  👉 이미 존재합니다: {filename}")
                continue

            print(f"⬇️ PDF 다운로드 시도: {filename} from {pdf_url}")
            
            # 5. PDF 파일 다운로드
            try:
                file_response = requests.get(pdf_url, stream=True, timeout=20)
                file_response.raise_for_status()

                # 6. 파일 저장
                with open(save_path, 'wb') as f:
                    # 큰 파일 다운로드를 위해 청크 방식으로 쓰기
                    for chunk in file_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"  ✅ 다운로드 완료: {filename}")
                pdf_count += 1
            
            except requests.exceptions.RequestException as e:
                print(f"  ❌ 다운로드 실패 ({filename}): {e}")
                
    if pdf_count > 0:
        print(f"\n🎉 총 {pdf_count}개의 PDF 파일을 '{download_dir}' 디렉터리에 다운로드했습니다.")
    else:
        print("\n⚠️ 웹페이지에서 PDF 파일을 찾을 수 없었습니다.")


if __name__ == '__main__':
    TARGET_URL = 'https://sites.google.com/cs.washington.edu/xai/schedule-reading'
    DOWNLOAD_FOLDER = 'CSE599_UW_ExplainableAI'
    
    download_pdfs_from_url(TARGET_URL, DOWNLOAD_FOLDER)
